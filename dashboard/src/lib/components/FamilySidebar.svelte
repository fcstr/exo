<script lang="ts">
  import FamilyLogos from "./FamilyLogos.svelte";

  type FamilySidebarProps = {
    families: string[];
    selectedFamily: string | null;
    hasFavorites: boolean;
    hasRecents: boolean;
    onSelect: (family: string | null) => void;
    platform?: string | null;
  };

  let {
    families,
    selectedFamily,
    hasFavorites,
    hasRecents,
    onSelect,
    platform = null,
  }: FamilySidebarProps = $props();

  // Family display names
  const familyNames: Record<string, string> = {
    favorites: "Favorites",
    recents: "Recent",
    huggingface: "Hub",
    huggingface_gguf: "GGUF",
    llama: "Meta",
    qwen: "Qwen",
    deepseek: "DeepSeek",
    "gpt-oss": "OpenAI",
    glm: "GLM",
    minimax: "MiniMax",
    kimi: "Kimi",
    flux: "FLUX",
    "qwen-image": "Qwen Img",
  };

  function getFamilyName(family: string): string {
    return (
      familyNames[family] || family.charAt(0).toUpperCase() + family.slice(1)
    );
  }
</script>

<div
  class="flex flex-col gap-1 py-2 px-1 border-r border-exo-yellow/10 bg-exo-medium-gray/30 min-w-[64px] overflow-y-auto scrollbar-hide"
>
  <!-- All models (no filter) -->
  <button
    type="button"
    onclick={() => onSelect(null)}
    class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
    null
      ? 'bg-exo-yellow/20 border-l-2 border-exo-yellow'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title="All models"
  >
    <svg
      class="w-5 h-5 {selectedFamily === null
        ? 'text-exo-yellow'
        : 'text-white/50 group-hover:text-white/70'}"
      viewBox="0 0 24 24"
      fill="currentColor"
    >
      <path
        d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z"
      />
    </svg>
    <span
      class="text-[9px] font-mono mt-0.5 {selectedFamily === null
        ? 'text-exo-yellow'
        : 'text-white/40 group-hover:text-white/60'}">All</span
    >
  </button>

  <!-- Favorites (only show if has favorites) -->
  {#if hasFavorites}
    <button
      type="button"
      onclick={() => onSelect("favorites")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'favorites'
        ? 'bg-exo-yellow/20 border-l-2 border-exo-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title="Show favorited models"
    >
      <FamilyLogos
        family="favorites"
        class={selectedFamily === "favorites"
          ? "text-amber-400"
          : "text-white/50 group-hover:text-amber-400/70"}
      />
      <span
        class="text-[9px] font-mono mt-0.5 {selectedFamily === 'favorites'
          ? 'text-amber-400'
          : 'text-white/40 group-hover:text-white/60'}">Faves</span
      >
    </button>
  {/if}

  <!-- Recent (only show if has recent models) -->
  {#if hasRecents}
    <button
      type="button"
      onclick={() => onSelect("recents")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'recents'
        ? 'bg-exo-yellow/20 border-l-2 border-exo-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title="Recently launched models"
    >
      <FamilyLogos
        family="recents"
        class={selectedFamily === "recents"
          ? "text-exo-yellow"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[9px] font-mono mt-0.5 {selectedFamily === 'recents'
          ? 'text-exo-yellow'
          : 'text-white/40 group-hover:text-white/60'}">Recent</span
      >
    </button>
  {/if}

  <!-- HuggingFace Hub (MLX) — hidden on llamacpp platform -->
  {#if platform !== "llamacpp"}
    <button
      type="button"
      onclick={() => onSelect("huggingface")}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      'huggingface'
        ? 'bg-orange-500/20 border-l-2 border-orange-400'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title="Browse and add MLX models from Hugging Face"
    >
      <FamilyLogos
        family="huggingface"
        class={selectedFamily === "huggingface"
          ? "text-orange-400"
          : "text-white/50 group-hover:text-orange-400/70"}
      />
      <span
        class="text-[9px] font-mono mt-0.5 {selectedFamily === 'huggingface'
          ? 'text-orange-400'
          : 'text-white/40 group-hover:text-white/60'}">Hub</span
      >
    </button>
  {/if}

  <!-- HuggingFace Hub (GGUF) -->
  <button
    type="button"
    onclick={() => onSelect("huggingface_gguf")}
    class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
    'huggingface_gguf'
      ? 'bg-green-500/20 border-l-2 border-green-400'
      : 'hover:bg-white/5 border-l-2 border-transparent'}"
    title="Browse and add GGUF models from Hugging Face"
  >
    <svg
      class="w-5 h-5 {selectedFamily === 'huggingface_gguf'
        ? 'text-green-400'
        : 'text-white/50 group-hover:text-green-400/70'}"
      viewBox="0 0 24 24"
      fill="currentColor"
    >
      <path d="M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-6 10H8v-2h6v2zm4-4H8v-2h10v2z" />
    </svg>
    <span
      class="text-[9px] font-mono mt-0.5 {selectedFamily === 'huggingface_gguf'
        ? 'text-green-400'
        : 'text-white/40 group-hover:text-white/60'}">GGUF</span
    >
  </button>

  <div class="h-px bg-exo-yellow/10 my-1"></div>

  <!-- Model families -->
  {#each families as family}
    <button
      type="button"
      onclick={() => onSelect(family)}
      class="group flex flex-col items-center justify-center p-2 rounded transition-all duration-200 cursor-pointer {selectedFamily ===
      family
        ? 'bg-exo-yellow/20 border-l-2 border-exo-yellow'
        : 'hover:bg-white/5 border-l-2 border-transparent'}"
      title={getFamilyName(family)}
    >
      <FamilyLogos
        {family}
        class={selectedFamily === family
          ? "text-exo-yellow"
          : "text-white/50 group-hover:text-white/70"}
      />
      <span
        class="text-[9px] font-mono mt-0.5 truncate max-w-full {selectedFamily ===
        family
          ? 'text-exo-yellow'
          : 'text-white/40 group-hover:text-white/60'}"
      >
        {getFamilyName(family)}
      </span>
    </button>
  {/each}
</div>
