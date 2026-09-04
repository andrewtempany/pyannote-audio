"""Tests for TICKET-03: the AMI dataset adapter.

harness/datasets.py doesn't exist yet at the time these tests are written --
the first run of this suite is expected to fail on import.

Fixtures live under tests/fixtures/ami/ (synthetic, not real AMI data):

- basic/{IHM,SDM}/test.{rttm,uem} -- the happy-path fixtures. IHM has two
  meetings (ES2002a, with a 3.0-5.0s overlap between spk1/spk2; ES2002b,
  no overlap). SDM has a single, differently-uri'd meeting (ES2003a) so
  that condition switching is unambiguous to assert on.
- missing_rttm_entry/IHM/test.{rttm,uem} -- the UEM references a second
  uri (ES2002z) that has no matching RTTM entry, for the missing-RTTM
  error-handling test.

Design decision this ticket asks to be made explicit (see also the
module docstring in harness/datasets.py once it exists): real AMI's IHM
condition has multiple raw per-participant headset-channel audio files
per meeting, but the ground-truth RTTM/UEM are inherently meeting-level
(one Annotation covering all speakers, independent of which physical mic
channel is later decoded). This adapter's uri space is therefore exactly
the set of uris present in a (condition, split)'s RTTM/UEM files -- one
`(uri, reference, uem)` per meeting, never per channel. Resolving a uri
to a specific audio path (e.g. a pre-mixed "Mix-Headset" file under IHM)
is out of scope for this adapter, which never touches audio at all.
"""

from pathlib import Path

import pytest
from pyannote.core import Annotation, Segment

from harness.config import HarnessConfig
from harness.datasets import AMIDatasetAdapter, DatasetError

FIXTURES = Path(__file__).parent / "fixtures" / "ami"


def _config(data_root, condition="IHM", cache_dir=None, tmp_path=None):
    return HarnessConfig.load(
        data_root=data_root,
        condition=condition,
        cache_dir=cache_dir or (tmp_path / "cache"),
    )


def test_iterates_expected_uris(tmp_path):
    config = _config(FIXTURES / "basic", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    uris = {uri for uri, _, _ in adapter}

    assert uris == {"ES2002a", "ES2002b"}


def test_reference_is_annotation_with_expected_labels(tmp_path):
    config = _config(FIXTURES / "basic", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    items = {uri: reference for uri, reference, _ in adapter}
    reference = items["ES2002a"]

    assert isinstance(reference, Annotation)
    assert reference.labels() == ["spk1", "spk2"]

    # compare segment/label content only -- not full Annotation equality,
    # which also compares internal track identifiers. Those come from
    # load_rttm's implementation (a pandas row index) and aren't part of
    # what this ticket asks the adapter to preserve.
    actual = sorted(
        (segment, label) for segment, _, label in reference.itertracks(yield_label=True)
    )
    expected = sorted(
        [
            (Segment(0.0, 5.0), "spk1"),
            (Segment(3.0, 7.0), "spk2"),
            (Segment(8.0, 10.0), "spk1"),
        ]
    )

    assert actual == expected


def test_uem_matches_uem_file(tmp_path):
    config = _config(FIXTURES / "basic", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    items = {uri: uem for uri, _, uem in adapter}

    assert items["ES2002a"].extent() == Segment(0.0, 10.0)
    assert items["ES2002b"].extent() == Segment(0.0, 5.0)


def test_default_condition_is_ihm(tmp_path):
    config = HarnessConfig.load(data_root=FIXTURES / "basic", cache_dir=tmp_path / "cache")
    adapter = AMIDatasetAdapter(config)

    uris = {uri for uri, _, _ in adapter}

    assert uris == {"ES2002a", "ES2002b"}


def test_condition_param_switches_to_sdm(tmp_path):
    config = _config(FIXTURES / "basic", condition="SDM", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    uris = {uri for uri, _, _ in adapter}

    assert uris == {"ES2003a"}


def test_ihm_multi_channel_handling(tmp_path):
    """Real AMI meetings under IHM have multiple raw headset-channel audio
    files, but the adapter's uri space comes only from the RTTM/UEM (which
    are meeting-level) -- so exactly one item per meeting is yielded, never
    one per channel, and no uri carries a channel suffix."""
    config = _config(FIXTURES / "basic", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    items = list(adapter)
    uris = [uri for uri, _, _ in items]

    assert len(items) == len(set(uris)) == 2
    assert all("Headset" not in uri and "." not in uri for uri in uris)


def test_missing_rttm_for_expected_uri_raises_clear_error(tmp_path):
    config = _config(FIXTURES / "missing_rttm_entry", tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    with pytest.raises(DatasetError) as exc_info:
        list(adapter)

    assert "ES2002z" in str(exc_info.value)


def test_missing_data_root_content_fails_clearly(tmp_path):
    empty_root = tmp_path / "empty_ami_root"
    empty_root.mkdir()
    config = _config(empty_root, tmp_path=tmp_path)
    adapter = AMIDatasetAdapter(config)

    with pytest.raises(DatasetError) as exc_info:
        list(adapter)

    message = str(exc_info.value)
    assert "IHM" in message
    assert "test" in message
