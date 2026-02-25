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
#   1. Installs system packages (build-essential, gcc-12/g++-12, cmake, curl, git,
#      nodejs, npm, libopenblas-dev, libjemalloc-dev)
#   2. Sets g++ default to version 12 (mlx JIT is incompatible with GCC 13+)
#   3. Installs uv (Python package/project manager)
#   4. Installs Rust via rustup (needed to compile exo_pyo3_bindings via maturin)
#   5. Clones exo from GitHub
#   6. Applies two Android/proot-specific patches to the Rust networking layer
#   7. Runs `uv sync` to build everything (Python + Rust)
#   8. Builds llama-cpp-python from source with -mcpu=native + OpenBLAS (GGUF inference)
#   9. Creates empty CUDA stub libs so mlx loads on Android (no NVIDIA driver)
#  10. Builds the Svelte dashboard with /usr/bin/npm (proot node, not Termux node)
#  11. Appends required environment variables to ~/.bashrc (incl. jemalloc + sshd)
#  12. Installs SSH pubkeys from GitHub (fcstr) and starts sshd on port 2222

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
    cmake \
    curl \
    git \
    nodejs \
    npm \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    libjemalloc-dev \
    ca-certificates \
    gcc-12 \
    g++-12

# mlx's JIT compiler generates C++ that uses 'typedef _Float128' which GCC 13+
# rejects (it became a built-in type). Force g++ 12 as the system default.
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 2>/dev/null || true
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12 2>/dev/null || true
update-alternatives --set gcc /usr/bin/gcc-12 2>/dev/null || true
update-alternatives --set g++ /usr/bin/g++-12 2>/dev/null || true
info "g++ set to $(g++ --version | head -1)"

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
    info "Cloning exo (proot-llamacpp branch) from fork..."
    if git clone --branch proot-llamacpp --single-branch \
            https://github.com/fcstr/exo.git "$EXO_DIR" 2>&1; then
        info "Cloned from proot-llamacpp fork — patches already applied"
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
# 7. Build llama-cpp-python from source with ARM64 optimizations + OpenBLAS
###############################################################################
#
# The PyPI wheel is compiled without -mcpu=native, missing hardware acceleration
# for dotprod, i8mm, bf16 instructions available on modern ARM SoCs (Cortex-X4,
# A720, etc). Rebuilding from source gives 2-4x speedup on quantized GGUF models.
# OpenBLAS accelerates matrix ops during prompt processing (prefill).

info "Building llama-cpp-python from source with native ARM optimizations + OpenBLAS..."
info "(This may take 3-5 minutes on a phone)"

CMAKE_ARGS="-DCMAKE_C_FLAGS='-mcpu=native -O3 -flto' -DCMAKE_CXX_FLAGS='-mcpu=native -O3 -flto' -DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  FORCE_CMAKE=1 \
  UV_LINK_MODE=copy \
  uv pip install --python "$EXO_DIR/.venv/bin/python" \
    llama-cpp-python==0.3.16 \
    --no-binary llama-cpp-python \
    --reinstall \
    --no-cache

info "llama-cpp-python built successfully with native ARM opts + OpenBLAS"

###############################################################################
# 8. Create CUDA stub libraries
###############################################################################
#
# mlx on Linux links libmlx.so against CUDA shared libraries even for CPU-only
# inference. On Android there is no NVIDIA GPU driver, so the dynamic linker
# fails to load libmlx.so. We create minimal empty stub .so files that satisfy
# the linker. mlx loads, detects no CUDA GPU, and falls back to CPU silently.

STUBS_DIR="$EXO_DIR/android-stubs"
mkdir -p "$STUBS_DIR"

# One-liner C source — valid empty shared library, exports nothing.
STUB_SRC="$(mktemp /tmp/stub_XXXXXX.c)"
echo "/* empty CUDA stub */" > "$STUB_SRC"

for lib in libcuda.so.1 libcublasLt.so.13 libnvrtc.so.13 libcudnn.so.9 libnccl.so.2; do
    if [[ ! -f "$STUBS_DIR/$lib" ]]; then
        gcc-12 -shared -fPIC -o "$STUBS_DIR/$lib" "$STUB_SRC" && info "created stub: $lib"
    else
        info "stub already exists: $lib"
    fi
done
rm -f "$STUB_SRC"
info "CUDA stubs created in $STUBS_DIR"

###############################################################################
# 8. Build the dashboard
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
EXO_ENV_MARKER="# exo / proot-llamacpp environment"

if grep -qF "$EXO_ENV_MARKER" "$BASHRC" 2>/dev/null; then
    info "$HOME/.bashrc already has exo env vars — skipping"
else
    info "Appending env vars to $BASHRC..."
    cat >> "$BASHRC" << ENVBLOCK

# exo / proot-llamacpp environment
export PATH="\$HOME/.cargo/bin:\$PATH"
export UV_LINK_MODE=copy
# mlx links against CUDA .so files even for CPU inference; stub libs satisfy
# the dynamic linker on Android where no NVIDIA driver exists.
export LD_LIBRARY_PATH="${EXO_DIR}/android-stubs:\$HOME/exo/.venv/lib/python3.13/site-packages/mlx_cuda_13.libs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# jemalloc reduces memory fragmentation under proot's ptrace overhead
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# Auto-start sshd on port 2222 (idempotent)
if ! pgrep -x sshd > /dev/null 2>&1; then
    /usr/sbin/sshd -p 2222
fi
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
# 10. SSH access (pubkey-only, port 2222)
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
echo "   source ~/.bashrc   # sets LD_LIBRARY_PATH, LD_PRELOAD, PATH"
echo "   cd $EXO_DIR"
echo "   uv run exo"
echo ""
echo " API will be available at: http://localhost:52415"
echo ""
echo " IMPORTANT — before running, do these in Termux (not proot):"
echo "   1. termux-wake-lock    # prevent Android killing proot"
echo "   2. Settings → Apps → Termux → Battery → Unrestricted"
echo "   3. Disable Samsung battery CPU limit (Settings → Battery)"
echo ""
echo " Note: mDNS peer discovery is disabled on Android/proot"
echo " (netlink multicast blocked by kernel). exo runs as a"
echo " single node — this is expected and harmless."
echo ""
echo " Inference engine: llama-cpp-python (GGUF) with OpenBLAS"
echo " Built with -mcpu=native for optimal ARM performance."
echo "============================================================"
