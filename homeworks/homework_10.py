from dataclasses import dataclass

import cv2
import numpy as np

DATA_DIR = "../data"


@dataclass
class Config:
    video_file = DATA_DIR + "/" + "book.mp4"
    image_file = DATA_DIR + "/" + "book.jpg"

    detector = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    dist_coeff = np.zeros(5)

    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1.0]])

    min_pnp_num = 100

    box_lower = np.array(
        [[30, 145, 0], [30, 200, 0], [200, 200, 0], [200, 145, 0]], dtype=np.float32
    )

    box_upper = np.array(
        [[30, 145, -50], [30, 200, -50], [200, 200, -50], [200, 145, -50]],
        dtype=np.float32,
    )


def main(cfg):
    image = cv2.imread(str(cfg.image_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_kp, im_desc = cfg.detector.detectAndCompute(gray, None)

    cap = cv2.VideoCapture(str(cfg.video_file))

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_desc = cfg.detector.detectAndCompute(gray, None)

        matches = cfg.matcher.match(im_desc, frame_desc)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([im_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        mask = mask.ravel().tolist()
        good_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

        if len(good_matches) < cfg.min_pnp_num:
            continue

        im_pts = []
        frame_pts = []

        for match in good_matches:
            im_point = im_kp[match.queryIdx].pt
            im_pts.append([im_point[0], im_point[1], 0])
            frame_pts.append(frame_kp[match.trainIdx].pt)

        im_pts = np.array(im_pts, dtype=np.float32)
        frame_pts = np.array(frame_pts, dtype=np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            im_pts, frame_pts, cfg.K, cfg.dist_coeff
        )

        image_box_lower, _ = cv2.projectPoints(
            cfg.box_lower, rvec, tvec, cfg.K, cfg.dist_coeff
        )

        image_box_upper, _ = cv2.projectPoints(
            cfg.box_upper, rvec, tvec, cfg.K, cfg.dist_coeff
        )

        show_image = frame.copy()

        cv2.polylines(show_image, [np.int32(image_box_lower)], True, (255, 0, 0), 2)

        cv2.polylines(show_image, [np.int32(image_box_upper)], True, (0, 0, 255), 2)

        cv2.imshow("Book", show_image)

        key = cv2.waitKey(10)

        if key == ord("q"):
            break
        elif key == ord(" "):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
