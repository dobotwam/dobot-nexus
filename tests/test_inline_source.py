import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dobot_nexus.data.lerobot.source import ImageLeRobotSource


def dataset(root, *, reverse_meta=False, split=False):
    (root/'meta').mkdir()
    (root/'data').mkdir()
    (root/'meta/info.json').write_text(json.dumps({'codebase_version': 'v2.1', 'fps': 10,
           'features': {'image': {'dtype': 'image'}}}))
    episodes = [{'episode_index': i, 'length': 2} for i in range(2)]
    if reverse_meta:
        episodes.reverse()
    (root/'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps, episodes)))
    (root/'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'pick'}))
    image = io.BytesIO()
    Image.new('RGB', (2, 2)).save(image, format='PNG')
    rows = [{'episode_index': i//2, 'frame_index': i%2, 'task_index': 0,
             'action': [float(i if i < 2 else 100+i)], 'state': [0.],
             'image': {'bytes': image.getvalue()}} for i in range(4)]
    chunks = [rows[:1], rows[1:3], rows[3:]] if split else [rows]
    for i, chunk in enumerate(chunks):
        pq.write_table(pa.Table.from_pylist(chunk), root/f'data/{i:03d}.parquet')
    return rows


def source(root):
    return ImageLeRobotSource(data_path=root, action_key='action', language_key=None, action_horizon=4)


class InlineSourceTest(unittest.TestCase):
    def test_metadata_order_does_not_cross_episodes_or_decode_future_images(self):
        for split in (False, True):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset(root, reverse_meta=True, split=split)
                reader = source(root)
                with patch.object(reader, '_row', wraps=reader._row) as row:
                    item = reader[0]
                    np.testing.assert_array_equal(item['action'][:, 0], [0, 1, 1, 1])
                    self.assertEqual(row.call_count, 1)
                    self.assertEqual(item['image'].shape, (2, 2, 3))
                np.testing.assert_array_equal(reader[2]['action'][:, 0], [102, 103, 103, 103])

    def test_invalid_frame_metadata_or_task_is_rejected_at_construction(self):
        for field, value in [('frame_index', 8), ('episode_index', 2), ('task_index', 99)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = dataset(root)
                rows[1][field] = value
                pq.write_table(pa.Table.from_pylist(rows), root/'data/000.parquet')
                with self.assertRaises(ValueError):
                    source(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root)
            (root/'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps, [
                {'episode_index': 0, 'length': 1}, {'episode_index': 1, 'length': 3}])))
            with self.assertRaisesRegex(ValueError, 'length'):
                source(root)
