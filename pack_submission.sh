#!/bin/bash
python3 task1.py --input_path images/t1 --output outputs/task1.png
python3 task2.py --input_path images/t2 --output outputs/task2.png --json ./task2.json

if find "images/Bonus1" -type f -iname '*.png' -print -quit | grep -q .; then
    python3 task2.py --input_path images/Bonus1 --output outputs/bonus1.png --json ./bonus1.json
fi

if find "images/Bonus2" -type f -iname '*.png' -print -quit | grep -q .; then
    python3 task2.py --input_path images/Bonus2 --output outputs/bonus2.png --json ./bonus2.json
fi

python3 utils.py --ubit $1