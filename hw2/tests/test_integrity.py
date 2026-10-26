import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gensim.models import FastText

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from newsvec.common import SentenceCorpus
from newsvec.prepare import clean_line, parse_articles
from newsvec.embeddings import document_vectors


class CorpusIntegrity(unittest.TestCase):
    def test_tones_unicode_and_punctuation(self):
        self.assertEqual(clean_line("应{ying4} 世界 经济 论坛 。 １２月 〓 创新 。"),
                         [["应", "世界", "经济", "论坛"], ["12月", "创新"]])

    def test_article_boundaries_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.txt"
            p.write_text("19980101-02-001-001 标题\n19980101-02-001-002 正文 新闻\n"
                         "19980101-02-002-001 另一 篇 新闻\n", encoding="utf-8")
            docs, _ = parse_articles(p)
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["page"], "02")
            self.assertEqual(docs[0]["tokens"], ["标题", "正文", "新闻"])
            self.assertTrue(all("19980101" not in w for d in docs for w in d["tokens"]))

    def test_stream_reiteration_and_fasttext_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sentences.txt"
            p.write_text("经济 市场 银行\n科技 创新 产业\n" * 5, encoding="utf-8")
            corpus = SentenceCorpus(p)
            self.assertEqual(list(corpus), list(corpus))
            model = FastText(vector_size=8, window=2, min_count=1, min_n=2, max_n=3,
                             bucket=128, workers=1, seed=42)
            model.build_vocab(corpus)
            model.train(corpus, total_examples=model.corpus_count, epochs=2)
            target = Path(tmp) / "model"
            model.save(str(target))
            restored = FastText.load(str(target))
            np.testing.assert_allclose(model.wv["经济"], restored.wv["经济"], rtol=0, atol=0)
            self.assertNotIn("经济创新", restored.wv.key_to_index)
            vec = restored.wv.get_vector("经济创新")
            self.assertTrue(np.isfinite(vec).all())
            self.assertGreater(np.linalg.norm(vec), 0)
            X, coverage = document_vectors([{"tokens": ["经济创新", "银行"]}], restored.wv)
            self.assertEqual(X.shape, (1, 8))
            self.assertEqual(coverage["represented_token_rate"], 1)


if __name__ == "__main__":
    unittest.main()
