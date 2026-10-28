Record #360 (ANVIL2, 1194 steps), the record before retrieval: `train_gpt.py` and `track_1_short/` as of 4ea6b937, with
three changes. It trains and validates on FineWeb-100B, as the retrieval record's baseline runs do. VAL_TOKEN_LOSSES
writes the final validation's per-token losses (`track_1_short/token_losses.py`;
src/retrieval_blog_post/token_losses.py runs it). REC360_BOS_FIX=1 waits for a shard's full BOS index when its partial
one runs out, where upstream ends the shard early (logged as BOS_RACE; on miso, runs that hit it land at 3.32-3.35).
