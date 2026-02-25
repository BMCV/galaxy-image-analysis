import argparse
import os

import cv2

from pathlib import Path
from typing import Union


def extract_frames(output_dir: Union[str, Path], video_path: Union[str, Path], start_time: int, end_time: int, convert_to_grey: str = "false") -> Path:
    """
    Extract frames from a video within a specified time range in seconds

    Parameters
    ----------
        video_path: Path to the input video file

        start_time: Start time in seconds

        end_time: End time in seconds

        output_dir: Directory where extracted frames will be saved

        convert_to_gray: Whether to convert frames to grayscale

    Returns:
    ---------
        Path to the directory containing the extracted frames.
    """

    try:
        video = cv2.VideoCapture(video_path)

        # get the video frames per second
        fps = video.get(cv2.CAP_PROP_FPS)
        print('Frames per second:', fps)
        # get the video total frames
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Total frames:', frame_count)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        print(f'Starting extracting from frame {start_frame} until {end_frame}...')

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = video.read()
            if not ret:
                break

            # Convert to single-channel grayscale
            output_path = os.path.join(output_dir, f"frame_{current_frame:05d}.tiff")
            if convert_to_grey == "true":
                cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            else:
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            current_frame += 1

        video.release()

        print('Extraction was successfully executed. Enjoy your frames.')

    except IOError:
        print("Cannot open video file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument('output_dir', help="Name of the output folder")
    parser.add_argument('-v', '--video_path', required=True, help="Path to the video to convert")
    parser.add_argument('-s', '--start_time', required=True, type=float, help="Start time in seconds")
    parser.add_argument('-e', '--end_time', required=True, type=float, help="End time in seconds")
    parser.add_argument('-c', '--convert_to_grey', required=False, type=str, help="Convert the file to grayscale")

    args = parser.parse_args()

    extract_frames(args.output_dir, video_path=args.video_path, start_time=args.start_time, end_time=args.end_time, convert_to_grey=args.convert_to_grey)
