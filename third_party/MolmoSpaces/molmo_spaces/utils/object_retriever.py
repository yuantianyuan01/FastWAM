import numpy as np
from molmospaces_resources import PickleLMDBMap

from molmo_spaces.env.object_manager import clip_sim, compute_text_clip
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.object_metadata import ObjectMeta


class ObjectRetriever:
    storage_path = ASSETS_DIR / ".lmdb" / "object_retriever"

    def __init__(self, sim_thres: float = 0.5, max_results: int = 50):
        self.thres = sim_thres
        self.max_results = max_results
        self.tk, self.ik, self.v = self.get_keys_values()

    def get_keys_values(self):
        # Take 7xx MiB of disk space. It's okay
        if not PickleLMDBMap.database_exists(self.storage_path):
            txt_keys, img_keys, values = [], [], []
            for uid, anno in ObjectMeta.annotation().items():
                values.append(uid)

                txt_clip = anno["clip_text_features"]
                txt_clip = (txt_clip / np.linalg.norm(txt_clip)).astype("float16")
                txt_keys.append(txt_clip)

                img_clip = anno["clip_img_features"]
                img_clip = (img_clip / np.linalg.norm(img_clip, axis=-1, keepdims=True)).astype(
                    "float16"
                )
                img_keys.append(img_clip)

            txt_keys = np.array(txt_keys)[:, None, :]
            img_keys = np.array(img_keys)
            values = np.array(values)

            PickleLMDBMap.from_dict(
                dict(txt_keys=txt_keys, img_keys=img_keys, values=values), self.storage_path
            )

            del txt_keys, img_keys, values

        map = PickleLMDBMap(self.storage_path)

        # keep them all in memory
        return map["txt_keys"], map["img_keys"], map["values"]

    def query(self, text):
        q = compute_text_clip(text)
        q = q / np.linalg.norm(q)

        sim = (
            clip_sim(self.ik, q, normalize=False)
            + 0.5 * clip_sim(self.tk, q, normalize=False, num_views=1)
        ).flatten()

        mask = sim >= self.thres
        rank = np.argsort(sim[mask])[::-1][: self.max_results]

        uids = self.v[mask][rank]
        sims = sim[mask][rank]

        return uids, sims


if __name__ == "__main__":
    r = ObjectRetriever()
    uids, sims = r.query("cellphone")
    for it, (uid, sim) in enumerate(zip(uids, sims)):
        anno = ObjectMeta.annotation(uid)
        print(
            f"{it} {sim=} uid={uid} obja={anno['isObjaverse']} split={anno['split']} cat=`{anno['category']}`:"
            f" {anno['description_short']['five_words']}"
        )

    print("DONE")
