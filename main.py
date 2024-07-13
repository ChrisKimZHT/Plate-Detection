import argparse
import datetime
import json
import os

import cv2
import hyperlpr3 as lpr3
from tqdm import tqdm


def draw_plate_on_image(img, box):
    x1, y1, x2, y2 = box
    h, w, _ = img.shape
    img_crop = img[y1:y2, x1:x2].copy()
    cv2.rectangle(img, (max(0, x1 - 5), max(0, y1 - 5)), (min(x2 + 5,
                  w - 1), min(y2 + 5, h - 1)), (0, 255, 0), 5, cv2.LINE_AA)
    return img, img_crop


def main(video_input: str, output_path: str):
    capture = cv2.VideoCapture(video_input)
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    box_id = 0
    result = {}

    for i in tqdm(range(total_frames)):
        ret, frame = capture.read()
        if not ret:
            break

        results = catcher(frame)

        for code, confidence, _, box in results:
            if (confidence < 0.85):
                continue

            if code not in result:
                result[code] = []

            result[code].append({
                'frame_id': i,
                'box_id': box_id,
                'confidence': confidence,
                'box': box
            })

            img_boxed, img_crop = draw_plate_on_image(frame, box)

            cv2.imwrite(os.path.join(
                output_path, f"boxed/{box_id}.jpg"), img_boxed)
            cv2.imwrite(os.path.join(
                output_path, f"crop/{box_id}.jpg"), img_crop)

            box_id += 1

    with open(os.path.join(output_path, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'total_frames': total_frames,
            'video_fps': video_fps,
            'total_boxes': box_id,
            'result': result
        }, f, indent=None, ensure_ascii=False)


if __name__ == '__main__':
    default_output_path = f'outputs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'

    paser = argparse.ArgumentParser()
    paser.add_argument('--input_video', type=str, default='video/test.mp4')
    paser.add_argument('--output_path', type=str, default=default_output_path)
    paser.add_argument('--log_file', type=str, default=None)
    args = paser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, 'boxed'))
        os.makedirs(os.path.join(args.output_path, 'crop'))

    if args.log_file is not None:
        log_file = open(f"{args.output_path}/{args.log_file}",
                        'w', encoding='utf-8')
        sys.stdout = log_file
        sys.stderr = log_file

    main(args.input_video, args.output_path)

    if args.log_file is not None:
        log_file.close()
