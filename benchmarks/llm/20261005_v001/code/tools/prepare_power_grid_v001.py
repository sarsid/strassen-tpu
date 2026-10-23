"""Build an exclusively created shape inventory and runnable campaign bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import build_power_midpoint_grid_v001 as geometry
import configure_power_grid_v001 as configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tiles = list(dict.fromkeys(configuration.CUBIC_TILES + configuration.STRASSEN_TILES))
    geometry.generate(args.output_dir, tiles=tiles)
    manifest = json.loads((args.output_dir / "sampled_shapes.json").read_text())
    base = json.loads((root / "configs/campaign_v1.json").read_text())
    campaign = configuration.build_config(base, manifest, "sampled_shapes.json")
    campaign_path = args.output_dir / "campaign_power_grid_v001.json"
    with campaign_path.open("x") as stream:
        json.dump(campaign, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (root / "configs/distributions_v1.json").open("rb") as source:
        with (args.output_dir / "distributions_v1.json").open("xb") as target:
            shutil.copyfileobj(source, target)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    with (args.output_dir / "bundle_manifest.json").open("x") as stream:
        json.dump({"sha256": hashes, "purpose": "Unmeasured preregistered sampled campaign"},
                  stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"exploratory": len(manifest["exploratory_shape_ids"]),
                      "holdout": len(manifest["holdout_shape_ids"]),
                      "campaign": str(campaign_path), "status": "prepared_unmeasured"}))


if __name__ == "__main__":
    main()
