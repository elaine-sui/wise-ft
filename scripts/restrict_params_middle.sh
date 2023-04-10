python src/wise_ft.py   \
    --train-dataset=ImageNet  \
    --epochs=10  \
    --lr=0.000015  \
    --batch-size=256  \
    --cache-dir=/pasteur/u/esui/wiseft/cache  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNet,ImageNetR,ImageNetA,ImageNetSketch  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --save=/pasteur/u/esui/wiseft/models/wiseft/ViTB32  \
    --data-location=/pasteur/u/esui/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --params_to_unfreeze=middle \
    --wandb