python3 tools/model_eval/eval_knn.py tasks/embeddings/emb_knn_vitb_imagenet.yaml \
meta/pretrained/emb_mtlir_vitb_tencentml_ep20.pth \
--nb_knn 20 --load_feat_path meta/test_infos
