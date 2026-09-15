import json
import tempfile
from pathlib import Path

from scene_discovery.common import file_sha256
from scene_discovery.feature_bank import DescriptorBank, Partition


def test_manifest_identifiers_remain_exact_strings():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "manifest.csv").write_text(
            "row_index,sample_id,sequence,split,timestamp,radar_index,lidar64_index,"
            "camera_front_index,lidar128_index,camera_rear_index\n"
            "0,sample,1,train,1643292946.710046076,00033,00001,00002,00001,00004\n",
            encoding="utf-8",
        )
        partition = Partition("1", "train", root, 1)
        row = partition.load_manifest().iloc[0]
        assert row["timestamp"] == "1643292946.710046076"
        assert row["radar_index"] == "00033"
        assert row["lidar64_index"] == "00001"
        assert row["camera_front_index"] == "00002"


def test_descriptor_bank_rejects_incomplete_and_stale_source():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        descriptor = root / "descriptor"
        source.mkdir()
        descriptor.mkdir()
        source_meta = source / "dataset_meta.json"
        source_meta.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        metadata = {
            "status": "complete",
            "source_feature_bank": str(source),
            "source_dataset_meta_sha256": file_sha256(source_meta),
            "build_signature": "signature",
            "splits": [],
            "modalities": [],
            "descriptor_kinds": [],
            "partitions": {},
        }
        (descriptor / "descriptor_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        try:
            DescriptorBank(descriptor)
        except RuntimeError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("Incomplete descriptor bank was accepted")

        (descriptor / "COMPLETE.json").write_text(
            json.dumps({"status": "complete", "build_signature": "signature"}),
            encoding="utf-8",
        )
        DescriptorBank(descriptor)
        source_meta.write_text(json.dumps({"status": "changed"}), encoding="utf-8")
        try:
            DescriptorBank(descriptor)
        except RuntimeError as error:
            assert "changed after descriptor creation" in str(error)
        else:
            raise AssertionError("Stale source metadata was accepted")
