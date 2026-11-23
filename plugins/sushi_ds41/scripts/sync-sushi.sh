#!/usr/bin/env bash
# Copies the DeepSeek-V4.1 slice of sushi into src/sushi/ in sushi's own folders (src/dsv41/, src/glm5/, src/exl3/), then
# applies this repo's edits to it (patches/*.patch). Usage: scripts/sync-sushi.sh <sushi checkout>
# The cut edges are ours and never overwritten: src/sushi/{model,transformer,status,scheduler,test_models,tokenizer,
# dflash,model_load,transient_peak,cooler_side,bills,gpu_mem,metal,pipeline_manifest,model_settings,mlx_util,trunk,
# prefill_plan,mtp_plan,sampler,test_note,chat,forward_refusal,request_media,server_state,test_support}.zig,
# src/sushi/dsv41/{repack,dsv41_state,pair_major_pack_test}.zig (only sushi's tests or Cooler server paths reach them),
# src/sushi/tensor/{route,matmul,attention,core}.zig, and src/sushi/arch/*.zig.
# scripts/sync-sushi.sh --check <sushi checkout> <ref>: does <ref> (a release tag or commit) change the slice since
# SUSHI_COMMIT? Exit 0 when the pin holds, 1 with the changed files when a sync is due.
set -euo pipefail
plugin="$(cd "$(dirname "$0")/.." && pwd -P)"
dst="$plugin/src/sushi"
dsv41="dsv41_borrow dsv41_config dsv41_engram dsv41_kernels dsv41_model dsv41_qat dsv41_scope dsv41_source dsv41_stream dsv41_vision dsv41_wide_bytes"
shared="arm_trial dflash_policy dspark_rows expert_bf16_kernels expert_drop expert_io expert_quant expert_stats expert_store expert_stream expert_usage fill_marks io_util load_clock log metal_minify mlx mlx_kernel_keys mlx_pages mtp_depth_bounds nax_gate owned_weights prefetch_plan round_cost spec_cost_gate test_expect vision_common weights_where"
cut="model transformer status scheduler test_models tokenizer dflash model_load transient_peak cooler_side bills gpu_mem metal pipeline_manifest model_settings mlx_util trunk prefill_plan mtp_plan sampler test_note chat forward_refusal request_media server_state test_support"
dsv41_cut="repack dsv41_state pair_major_pack_test"

slice_paths() {
  local p
  echo src/exl3; echo src/dsv41/fixtures/dsv41_engram_token_map.bin; echo src/glm5/glm_stream_fixture.zig; echo src/tensor/pool_class.zig
  for p in $dsv41; do echo "src/dsv41/$p.zig"; done
  for p in $shared; do echo "src/$p.zig"; done
}

if [ "${1:-}" = "--check" ]; then
  repo="${2:?sushi checkout}"; ref="${3:?release tag or commit}"
  pin="$(tr -d '[:space:]' < "$dst/SUSHI_COMMIT")"
  paths=()
  while read -r p; do
    # A path the ref does not have would match nothing and read as "pin holds": the slice moved, so a sync is due.
    git -C "$repo" cat-file -e "$ref:$p" 2>/dev/null || { echo "sync due: $ref has no $p (the slice moved)"; exit 1; }
    paths+=("$p")
  done < <(slice_paths)
  changed="$(git -C "$repo" diff --name-only "$pin" "$ref" -- "${paths[@]}")"
  if [ -z "$changed" ]; then echo "pin holds: $ref changes nothing in the slice since ${pin:0:8}"; exit 0; fi
  echo "sync due: $ref changes the slice since ${pin:0:8}:"; echo "$changed" | sed 's/^/  /'; exit 1
fi

src="${1:?sushi checkout}/src"
# Everything but the cut edges goes first, so a file sushi dropped does not linger.
keep=()
for f in $cut; do keep+=(! -name "$f.zig"); done
find "$dst" -maxdepth 1 -type f -name '*.zig' "${keep[@]}" -delete
keep41=()
for f in $dsv41_cut; do keep41+=(! -name "$f.zig"); done
find "$dst/dsv41" -maxdepth 1 -type f -name '*.zig' "${keep41[@]}" -delete
rm -rf "$dst/dsv41/fixtures" "$dst/glm5" "$dst/exl3" "$dst/fixtures" "$dst/tensor/pool_class.zig"
mkdir -p "$dst/dsv41/fixtures" "$dst/glm5" "$dst/exl3" "$dst/tensor" "$dst/arch"
for f in $dsv41; do cp "$src/dsv41/$f.zig" "$dst/dsv41/$f.zig"; done
for f in $shared; do cp "$src/$f.zig" "$dst/$f.zig"; done
cp "$src/tensor/pool_class.zig" "$dst/tensor/pool_class.zig"
cp "$src/glm5/glm_stream_fixture.zig" "$dst/glm5/"
cp "$src"/exl3/*.zig "$dst/exl3/"
cp "$src/dsv41/fixtures/dsv41_engram_token_map.bin" "$dst/dsv41/fixtures/"
# patches/*.patch, in name order (paths relative to the plugin). `git apply` is all or nothing: a patch that no longer
# applies to the new slice stops the sync with the files it names.
top="$(git -C "$plugin" rev-parse --show-toplevel)"
for p in "$plugin"/patches/*.patch; do
  [ -e "$p" ] || continue
  git -C "$top" apply --directory="${plugin#"$top"/}" "$p" || { echo "sync-sushi: $(basename "$p") does not apply; redo it on the new slice" >&2; exit 1; }
done
echo "$(git -C "${1}" rev-parse HEAD)" > "$dst/SUSHI_COMMIT"
