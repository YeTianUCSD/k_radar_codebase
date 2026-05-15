import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ASF cropped camera images from undistorted images.')
    parser.add_argument('--src-root', default='/home/code/hyperradar/dataset/k_radar_asf_preprocessed/kradar_imgs/undistorted')
    parser.add_argument('--dst-root', default='/home/code/hyperradar/dataset/k_radar_asf_preprocessed/kradar_imgs/cropped')
    parser.add_argument('--seqs', nargs='+', default=['1'])
    parser.add_argument('--cams', nargs='+', default=['front0'])
    parser.add_argument('--resize', type=float, default=0.7)
    parser.add_argument('--crop', type=int, nargs=4, default=[96, 170, 800, 426])
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def crop_one_image(src_path, dst_path, resize, crop, overwrite=False):
    if dst_path.exists() and not overwrite:
        return False

    with Image.open(src_path) as img:
        resized_size = (int(img.width * resize), int(img.height * resize))
        img = img.resize(resized_size, Image.BILINEAR)
        img = img.crop(tuple(crop))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
    return True


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    total = 0
    written = 0
    for seq in args.seqs:
        for cam in args.cams:
            src_dir = src_root / seq / cam
            dst_dir = dst_root / seq / cam
            src_paths = sorted(src_dir.glob('*.png'))
            if not src_paths:
                raise FileNotFoundError(f'No PNG files found in {src_dir}')

            for src_path in tqdm(src_paths, desc=f'seq{seq}/{cam}'):
                dst_path = dst_dir / src_path.name
                total += 1
                if crop_one_image(src_path, dst_path, args.resize, args.crop, args.overwrite):
                    written += 1

    print(f'processed={total}, written={written}, skipped={total-written}')


if __name__ == '__main__':
    main()
