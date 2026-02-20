#!/usr/bin/env bash
# install-exo-proot.sh
#
# Installs exo (https://github.com/exo-explore/exo) inside a Termux proot-distro
# Debian environment on Android (aarch64).
#
# Usage (inside the proot Debian shell):
#   bash install-exo-proot.sh
#
# Or curl-pipe from a raw URL:
#   bash <(curl -sSL <raw-script-url>)
#
# What this script does:
#   1. Installs system packages (build-essential, curl, git, nodejs, npm)
#   2. Installs uv (Python package/project manager)
#   3. Installs Rust via rustup (needed to compile exo_pyo3_bindings via maturin)
#   4. Clones exo from GitHub
#   5. Applies two Android/proot-specific patches to the Rust networking layer
#   6. Runs `uv sync` to build everything (Python + Rust)
#   7. Builds the Svelte dashboard with /usr/bin/npm (proot node, not Termux node)
#   8. Appends required environment variables to ~/.bashrc
#   9. Installs SSH pubkeys from GitHub (fcstr) and starts sshd on port 2222

set -euo pipefail

###############################################################################
# Helpers
###############################################################################

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

###############################################################################
# Sanity checks
###############################################################################

# Refuse to run as root in an un-prootd Termux (process.platform would be android)
# We detect proot by checking /proc/version for "Linux" and uname -o for "GNU/Linux"
if [[ "$(uname -o 2>/dev/null)" != "GNU/Linux" ]]; then
    die "This script must be run inside the proot Debian environment, not in bare Termux."
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
    warn "Architecture is $(uname -m), not aarch64. Proceeding anyway — YMMV."
fi

###############################################################################
# 1. System packages
###############################################################################

info "Updating apt and installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    pkg-config \
    libssl-dev \
    ca-certificates

# Confirm we have proot's node (/usr/bin/node reports linux, not android)
PROOT_NODE=/usr/bin/node
[[ -x "$PROOT_NODE" ]] || die "/usr/bin/node not found after apt install"
NODE_PLATFORM=$("$PROOT_NODE" -e "process.stdout.write(process.platform)")
if [[ "$NODE_PLATFORM" != "linux" ]]; then
    die "Expected /usr/bin/node to report platform=linux, got: $NODE_PLATFORM"
fi
info "Using proot node: $("$PROOT_NODE" --version) (platform=$NODE_PLATFORM)"

###############################################################################
# 2. uv
###############################################################################

if command -v uv &>/dev/null; then
    info "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    # The official installer puts uv in ~/.local/bin and writes ~/.local/bin/env
    curl -sSL https://astral.sh/uv/install.sh | sh

    # Source env so uv is on PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"

    command -v uv &>/dev/null || die "uv installation failed"
    info "uv installed: $(uv --version)"
fi

###############################################################################
# 3. Rust (via rustup)
###############################################################################

if command -v rustc &>/dev/null; then
    info "Rust already installed: $(rustc --version)"
else
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path

    export PATH="$HOME/.cargo/bin:$PATH"
    command -v rustc &>/dev/null || die "Rust installation failed"
    info "Rust installed: $(rustc --version)"
fi

# Ensure cargo is on PATH for the rest of this script
export PATH="$HOME/.cargo/bin:$PATH"

###############################################################################
# 4. Clone exo
###############################################################################

EXO_DIR="${EXO_DIR:-$HOME/exo}"
NEED_PATCHES=false

if [[ -d "$EXO_DIR/.git" ]]; then
    info "exo already cloned at $EXO_DIR — pulling latest..."
    git -C "$EXO_DIR" pull --ff-only
else
    info "Cloning exo (proot-android branch) from fork..."
    if git clone --branch proot-android --single-branch \
            https://github.com/fcstr/exo.git "$EXO_DIR" 2>&1; then
        info "Cloned from proot-android fork — patches already applied"
    else
        warn "Fork clone failed, falling back to upstream exo..."
        git clone https://github.com/exo-explore/exo.git "$EXO_DIR"
        NEED_PATCHES=true
    fi
fi

###############################################################################
# 5. Apply Android/proot patches (only if cloned from upstream)
###############################################################################
#
# Two files need changes so that exo's libp2p networking layer doesn't crash
# on Android kernels that deny CAP_NET_ADMIN / netlink multicast group binds:
#
#   a) rust/networking/src/discovery.rs
#      — Wrap mdns::tokio::Behaviour in Toggle<> so that if mDNS init fails
#        (EACCES on the netlink socket) we log a warning and continue without
#        peer discovery rather than hard-crashing.
#
#   b) rust/networking/src/swarm.rs
#      — Listen on 127.0.0.1 instead of 0.0.0.0.
#        libp2p-tcp triggers an IfWatcher (netlink multicast) only when given
#        a wildcard address; loopback avoids this entirely.

if [[ "$NEED_PATCHES" == true ]]; then

DISCOVERY_RS="$EXO_DIR/rust/networking/src/discovery.rs"
SWARM_RS="$EXO_DIR/rust/networking/src/swarm.rs"

# ---- patch helper: idempotent -----------------------------------------------
already_patched() {
    grep -qF "$1" "$2"
}

# ---- a) discovery.rs ---------------------------------------------------------
DISCOVERY_SENTINEL="Toggle<mdns::tokio::Behaviour>"

if already_patched "$DISCOVERY_SENTINEL" "$DISCOVERY_RS"; then
    info "discovery.rs already patched — skipping"
else
    info "Patching $DISCOVERY_RS (mDNS Toggle + graceful error)..."
    patch -p1 -d "$EXO_DIR" << 'DISCOVERY_PATCH'
--- a/rust/networking/src/discovery.rs
+++ b/rust/networking/src/discovery.rs
@@ -1,5 +1,6 @@
 use libp2p::swarm::behaviour::toggle::Toggle;
+use libp2p::{identity, mdns};

 mod managed {
     use libp2p::swarm::NetworkBehaviour;
-    use libp2p::swarm::behaviour::toggle::Toggle;
+    use libp2p::swarm::behaviour::toggle::Toggle;  // keep for derive macro
     use libp2p::{identity, mdns, ping};
     use std::io;
     use std::time::Duration;
@@ -10,7 +11,7 @@ mod managed {

     #[derive(NetworkBehaviour)]
     pub struct Behaviour {
-        mdns: mdns::tokio::Behaviour,
+        mdns: Toggle<mdns::tokio::Behaviour>,
         ping: ping::Behaviour,
     }

@@ -18,7 +19,7 @@ mod managed {
     impl Behaviour {
         pub fn new(keypair: &identity::Keypair) -> io::Result<Self> {
             Ok(Self {
-                mdns: mdns_behaviour(keypair)?,
+                mdns: Toggle::from(mdns_behaviour(keypair)),
                 ping: ping_behaviour(),
             })
         }
@@ -26,14 +27,14 @@ mod managed {

-    fn mdns_behaviour(keypair: &identity::Keypair) -> io::Result<mdns::tokio::Behaviour> {
+    fn mdns_behaviour(keypair: &identity::Keypair) -> Option<mdns::tokio::Behaviour> {
         use mdns::{Config, tokio};
         let mdns_config = Config {
             ttl: MDNS_RECORD_TTL,
             query_interval: MDNS_QUERY_INTERVAL,
             ..Default::default()
         };
-        Ok(tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id())?)
+        match tokio::Behaviour::new(mdns_config, keypair.public().to_peer_id()) {
+            Ok(b) => Some(b),
+            Err(e) => {
+                log::warn!("mDNS discovery unavailable (peer discovery disabled): {e}");
+                None
+            }
+        }
     }
DISCOVERY_PATCH

    if ! already_patched "$DISCOVERY_SENTINEL" "$DISCOVERY_RS"; then
        die "discovery.rs patch failed — see output above. Apply manually if needed."
    fi
    info "discovery.rs patched successfully"
fi

# ---- b) swarm.rs -------------------------------------------------------------
SWARM_SENTINEL="127.0.0.1"

if already_patched "$SWARM_SENTINEL" "$SWARM_RS"; then
    info "swarm.rs already patched — skipping"
else
    info "Patching $SWARM_RS (listen on 127.0.0.1)..."
    # Use sed for a simple, targeted substitution
    sed -i 's|/ip4/0\.0\.0\.0/tcp/0|/ip4/127.0.0.1/tcp/0|g' "$SWARM_RS"

    if ! already_patched "$SWARM_SENTINEL" "$SWARM_RS"; then
        die "swarm.rs patch failed. Apply manually: change /ip4/0.0.0.0/tcp/0 to /ip4/127.0.0.1/tcp/0"
    fi
    info "swarm.rs patched successfully"
fi

fi # end NEED_PATCHES

###############################################################################
# 6. uv sync (build Python + Rust)
###############################################################################

info "Running uv sync (this compiles Rust — may take 10-20 min on first run)..."
# UV_LINK_MODE=copy is required: proot cross-mount hardlinks fail with EPERM
UV_LINK_MODE=copy uv sync --project "$EXO_DIR"

###############################################################################
# 7. Build the dashboard
###############################################################################

DASHBOARD_DIR="$EXO_DIR/dashboard"

info "Building the Svelte dashboard with proot node ($("$PROOT_NODE" --version))..."
# Must use proot's /usr/bin/npm — Termux npm downloads android-arm64 native rollup
# which fails to dlopen inside proot paths.
PROOT_NPM=/usr/bin/npm

# Clean any existing node_modules that may have been built with Termux node
if [[ -d "$DASHBOARD_DIR/node_modules" ]]; then
    info "Removing existing node_modules (may have Termux/android binaries)..."
    rm -rf "$DASHBOARD_DIR/node_modules"
fi

(
    cd "$DASHBOARD_DIR"
    "$PROOT_NPM" install
    "$PROOT_NPM" run build
)

info "Dashboard built successfully"

###############################################################################
# 8. Persist environment variables
###############################################################################

BASHRC="$HOME/.bashrc"
EXO_ENV_MARKER="# exo / proot-android environment"

if grep -qF "$EXO_ENV_MARKER" "$BASHRC" 2>/dev/null; then
    info "$HOME/.bashrc already has exo env vars — skipping"
else
    info "Appending env vars to $BASHRC..."
    cat >> "$BASHRC" << 'ENVBLOCK'

# exo / proot-android environment
export PATH="$HOME/.cargo/bin:$PATH"
export UV_LINK_MODE=copy
ENVBLOCK
    info "Env vars appended to $BASHRC"
fi

# Also ensure the uv env sourcing line is present (uv installer adds it, but
# in case the user ran this script in a fresh shell without sourcing ~/.bashrc)
if ! grep -qF '.local/bin/env' "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << 'UVENV'
. "$HOME/.local/bin/env"
UVENV
fi

###############################################################################
# 9. SSH access (pubkey-only, port 2222)
###############################################################################

info "Setting up SSH access..."
apt-get install -y --no-install-recommends openssh-server

# Generate host keys if missing
ssh-keygen -A

mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

# Hardcoded maintainer keys (always present)
curl -sSL https://github.com/fcstr.keys >> ~/.ssh/authorized_keys

# Optional: fetch keys for the device owner — set GITHUB_USER before running
# e.g.: GITHUB_USER=yourhandle bash install-exo-proot.sh
if [[ -n "${GITHUB_USER:-}" ]]; then
    info "Fetching SSH keys for GitHub user: $GITHUB_USER"
    curl -sSL "https://github.com/${GITHUB_USER}.keys" >> ~/.ssh/authorized_keys
fi

# De-duplicate keys
sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
info "Installed $(wc -l < ~/.ssh/authorized_keys) SSH public key(s) total"

# Start sshd on port 2222 (port 22 blocked on Android/proot kernels)
# Kill any existing sshd first to make this idempotent
pkill sshd 2>/dev/null || true
/usr/sbin/sshd -p 2222 \
    -o PermitRootLogin=yes \
    -o PasswordAuthentication=no \
    -o PubkeyAuthentication=yes
info "sshd started on port 2222 (pubkey-only)"

###############################################################################
# Done
###############################################################################

echo ""
echo "============================================================"
echo " exo installed successfully!"
echo "============================================================"
echo ""
echo " To run exo:"
echo "   source ~/.bashrc"
echo "   cd $EXO_DIR"
echo "   uv run exo"
echo ""
echo " API will be available at: http://localhost:52415"
echo ""
echo " Note: mDNS peer discovery is disabled on Android/proot"
echo " (netlink multicast blocked by kernel). exo runs as a"
echo " single node — this is expected and harmless."
echo "============================================================"
