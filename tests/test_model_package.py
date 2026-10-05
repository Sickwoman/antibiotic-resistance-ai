"""The frozen model's package and installer (src/model_package.py, scripts/install_model.py, scripts/package_model.py).

Most checks build packages from fake bytes, with the pin swapped for the duration of the test, so they need no model
and never touch `models/`. The checks with the real bundle are skipped when `models/` is absent (CI).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from src import model_package as mp

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / mp.TARGET
needs_model = pytest.mark.skipif(not REAL.is_file(), reason="the frozen model bundle is not in this checkout")
FAKE = b"not a real model: stands in for the bundle's bytes in tests\n" * 100


@pytest.fixture
def pinned_fake(monkeypatch):
    monkeypatch.setattr(mp, "FROZEN_BUNDLE_SHA256", hashlib.sha256(FAKE).hexdigest())
    monkeypatch.setattr(mp, "FROZEN_BUNDLE_SIZE", len(FAKE))
    return FAKE


def package(tmp_path: Path, bundle: bytes = FAKE, **override: bytes) -> Path:
    members = {mp.BUNDLE_NAME: bundle, mp.SIDECAR_NAME: mp.sidecar_text().encode(),
               "MANIFEST.json": json.dumps({"bundle": {"sha256": mp.FROZEN_BUNDLE_SHA256}}).encode(),
               "MODEL_CARD.md": b"card\n", "NOTICE.md": b"notice\n", **override}
    path = tmp_path / "package.zip"
    path.write_bytes(mp.zip_members(members))
    return path


def test_the_pin_is_the_value_published_in_the_repository():
    assert mp.FROZEN_BUNDLE_SHA256 in (ROOT / "docs/reproduction_guide.md").read_text(encoding="utf-8")
    assert mp.FROZEN_BUNDLE_SHA256 in (ROOT / "docs/research_demo_brief.md").read_text(encoding="utf-8")
    assert mp.FROZEN_BUNDLE_SIZE == 636_912


def test_the_install_target_is_where_the_unchanged_cli_looks():
    spec = importlib.util.spec_from_file_location("predict_spectrum", ROOT / "scripts/predict_spectrum.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert Path(cli.default_model(cli.load_config(None))) == ROOT / mp.TARGET


def test_nothing_in_the_installer_can_deserialise_the_bundle():
    for path in ("src/model_package.py", "scripts/install_model.py"):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not imported & {"joblib", "pickle", "dill", "cloudpickle", "src.predict", "urllib.request",
                               "http.client", "requests", "httpx"}, path


def test_a_build_is_reproducible_and_holds_exactly_the_five_members(pinned_fake):
    first = mp.build(FAKE, {"bundle": {"sha256": mp.FROZEN_BUNDLE_SHA256}}, "card\n", "notice\n")
    assert first == mp.build(FAKE, {"bundle": {"sha256": mp.FROZEN_BUNDLE_SHA256}}, "card\n", "notice\n")
    with zipfile.ZipFile(__import__("io").BytesIO(first)) as archive:
        infos = archive.infolist()
        assert [i.filename for i in infos] == list(mp.MEMBERS)
        assert all(i.compress_type == zipfile.ZIP_STORED and i.date_time == mp.ZIP_TIME for i in infos)
        assert archive.read(mp.BUNDLE_NAME) == FAKE


def test_only_the_frozen_bundle_can_be_packaged():
    with pytest.raises(mp.PackageError):
        mp.build(FAKE, {"bundle": {"sha256": mp.FROZEN_BUNDLE_SHA256}}, "card\n", "notice\n")


def test_a_verified_package_installs_the_bundle_and_its_checksum(pinned_fake, tmp_path):
    target = tmp_path / "models" / mp.BUNDLE_NAME
    result = mp.install(package(tmp_path), target=target)
    assert result.changed and target.read_bytes() == FAKE
    assert target.with_name(target.name + ".sha256").read_text() == mp.sidecar_text()
    again = mp.install(package(tmp_path), target=target)
    assert not again.changed and "already installed" in again.message
    assert not list(target.parent.glob(".installing-*"))


def test_an_extracted_folder_installs_too(pinned_fake, tmp_path):
    folder = tmp_path / "extracted"
    with zipfile.ZipFile(package(tmp_path)) as archive:
        archive.extractall(folder)
    target = tmp_path / "models" / mp.BUNDLE_NAME
    assert mp.install(folder, target=target).changed and target.read_bytes() == FAKE


def test_a_different_bundle_is_refused_and_nothing_is_written(pinned_fake, tmp_path):
    target = tmp_path / "models" / mp.BUNDLE_NAME
    tampered = bytearray(FAKE)
    tampered[10] ^= 0x01
    with pytest.raises(mp.PackageError) as info:
        mp.install(package(tmp_path, bytes(tampered)), target=target)
    assert info.value.code == mp.VERIFICATION_FAILED and "Refused" in str(info.value)
    assert not target.parent.exists()


def test_a_damaged_archive_is_refused(pinned_fake, tmp_path):
    path = package(tmp_path)
    data = bytearray(path.read_bytes())
    data[data.index(FAKE[:20]) + 5] ^= 0x01                 # inside the stored bundle: the CRC no longer matches
    path.write_bytes(bytes(data))
    with pytest.raises(mp.PackageError) as info:
        mp.install(path, target=tmp_path / "models" / mp.BUNDLE_NAME)
    assert info.value.code == mp.VERIFICATION_FAILED and "damaged" in str(info.value)


def test_the_packages_own_files_can_refuse_but_never_accept(pinned_fake, tmp_path):
    other = "0" * 64
    for override in ({mp.SIDECAR_NAME: f"{other}  {mp.BUNDLE_NAME}\n".encode()},
                     {"MANIFEST.json": json.dumps({"bundle": {"sha256": other}}).encode()},
                     {"MANIFEST.json": b"{not json"}):
        with pytest.raises(mp.PackageError):
            mp.install(package(tmp_path, **override), target=tmp_path / "models" / mp.BUNDLE_NAME)
    tampered = FAKE + b"x"
    honest_about_itself = {mp.SIDECAR_NAME: f"{hashlib.sha256(tampered).hexdigest()}  x\n".encode()}
    with pytest.raises(mp.PackageError):
        mp.install(package(tmp_path, tampered, **honest_about_itself), target=tmp_path / "m" / mp.BUNDLE_NAME)


def test_missing_packages_urls_and_strangers_are_refused(pinned_fake, tmp_path):
    for bad in (tmp_path / "absent.zip", "https://example.org/model.zip"):
        with pytest.raises(mp.PackageError) as info:
            mp.install(bad, target=tmp_path / "models" / mp.BUNDLE_NAME)
        assert info.value.code == mp.VERIFICATION_FAILED
    with_extra = package(tmp_path)
    with zipfile.ZipFile(with_extra, "a") as archive:
        archive.writestr("extra.py", "print('x')")
    with pytest.raises(mp.PackageError, match="unexpected members"):
        mp.install(with_extra, target=tmp_path / "models" / mp.BUNDLE_NAME)
    no_bundle = tmp_path / "empty.zip"
    with zipfile.ZipFile(no_bundle, "w") as archive:
        archive.writestr("NOTICE.md", "x")
    with pytest.raises(mp.PackageError, match="holds no"):
        mp.install(no_bundle, target=tmp_path / "models" / mp.BUNDLE_NAME)


def test_a_different_existing_model_or_checksum_is_never_replaced(pinned_fake, tmp_path):
    target = tmp_path / "models" / mp.BUNDLE_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(b"someone else's model")
    with pytest.raises(mp.PackageError) as info:
        mp.install(package(tmp_path), target=target)
    assert info.value.code == mp.REFUSED_OVERWRITE and target.read_bytes() == b"someone else's model"
    target.unlink()
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text("f" * 64 + "  best_random.joblib\n")
    with pytest.raises(mp.PackageError) as info:
        mp.install(package(tmp_path), target=target)
    assert info.value.code == mp.REFUSED_OVERWRITE and not target.exists()


def test_the_command_reports_failures_with_their_exit_codes(pinned_fake, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("install_model", ROOT / "scripts/install_model.py")
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    target = tmp_path / "models" / mp.BUNDLE_NAME
    assert command.main([str(tmp_path / "absent.zip"), "--target", str(target)]) == mp.VERIFICATION_FAILED
    assert "Not installed" in capsys.readouterr().err
    assert command.main([str(package(tmp_path)), "--target", str(target)]) == 0
    assert "Installed the frozen model" in capsys.readouterr().out


def test_the_committed_release_documents_name_the_frozen_bundle():
    docs = ROOT / "docs/release/model-v0.4.0"
    manifest = json.loads((docs / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["bundle"] == {**manifest["bundle"], "sha256": mp.FROZEN_BUNDLE_SHA256,
                                  "bytes": mp.FROZEN_BUNDLE_SIZE}
    assert manifest["evaluation"]["external_marisma_2024"]["zone_r_or_i"] == 18
    assets = json.loads((docs / "ASSETS.json").read_text(encoding="utf-8"))
    assert assets["assets"][0]["name"] == f"{mp.PACKAGE_NAME}.zip"
    notice = (docs / "NOTICE.md").read_text(encoding="utf-8")
    assert "take effect only when the copyright holder" in notice and mp.FROZEN_BUNDLE_SHA256 in notice


@needs_model
def test_the_real_package_is_reproducible_and_installs_without_touching_models(tmp_path):
    spec = importlib.util.spec_from_file_location("package_model", ROOT / "scripts/package_model.py")
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    before = hashlib.sha256(REAL.read_bytes()).hexdigest()
    committed = {n: (ROOT / packager.DOCS / n).read_bytes() for n in ("MANIFEST.json", "MODEL_CARD.md", "NOTICE.md",
                                                                      "ASSETS.json", "RELEASE_NOTES.md")}
    assert packager.main(["--out", str(tmp_path / "dist")]) == 0
    after_docs = {n: (ROOT / packager.DOCS / n).read_bytes() for n in committed}
    assert after_docs == committed                              # the committed review copies are what it renders
    built = tmp_path / "dist" / f"{mp.PACKAGE_NAME}.zip"
    assert hashlib.sha256(built.read_bytes()).hexdigest() == json.loads(committed["ASSETS.json"])["assets"][0]["sha256"]
    result = mp.install(built, target=tmp_path / "models" / mp.BUNDLE_NAME)
    assert result.changed and hashlib.sha256(result.target.read_bytes()).hexdigest() == mp.FROZEN_BUNDLE_SHA256
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == before


def test_the_publication_record_quotes_the_built_assets_exactly():
    docs = ROOT / "docs/release/model-v0.4.0"
    assets = json.loads((docs / "ASSETS.json").read_text(encoding="utf-8"))["assets"]
    text = "".join((docs / name).read_text(encoding="utf-8") for name in ("PUBLICATION.md", "RELEASE_NOTES.md"))
    for asset in assets:
        assert asset["name"] in text and asset["sha256"] in text and f"{asset['bytes']:,}" in text
    assert "--latest=false" in text and "model-ecoli-ciprofloxacin-v0.4.0" in text
