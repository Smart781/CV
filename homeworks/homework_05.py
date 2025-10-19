from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn


class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    @torch.inference_mode()
    def forward(self, image):
        x = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) / 255.0
        x = x[None, None]

        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        coarse_desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # Postprocessing
        semi = semi.cpu().numpy().squeeze()
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + 0.00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]

        h, w = nodust.shape[1:3]
        H, W = 8 * h, 8 * w

        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [h, w, 8, 8])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [h * 8, w * 8])
        xs, ys = np.where(heatmap >= 0.015)  # Confidence threshold.

        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]

        pts, _ = self.nms_fast(pts, H, W, dist_thresh=4)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.

        # Interpolate into descriptor map using 2D point locations.
        samp_pts = torch.from_numpy(pts[:2, :].copy())
        samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.0)) - 1.0
        samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.0)) - 1.0
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        desc = nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
        D = coarse_desc.shape[1]
        desc = desc.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        keypoints = pts[:2].T.astype(int)
        descriptors = desc.T.astype(np.float32)

        return keypoints, descriptors

    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W)).astype(int)
        inds = np.zeros((H, W)).astype(int)

        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.

        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)

        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)

        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i

        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode="constant")

        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0

        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)

            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad : pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1

        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]

        return out, out_inds


def desc_distances(cfg, image1, image2):
    superpoint_weights = cfg.superpoint_weights

    net = SuperPointNet()
    net.load_state_dict(torch.load(superpoint_weights))

    pts1, des1 = net(image1)
    pts2, des2 = net(image2)

    dots = np.dot(des2, des1.T)
    D = 2 * (1 - np.clip(dots, -1, 1))

    return D, pts1, pts2


def Cross_Check_Matching(cfg, D, image1, image2, pts1, pts2):
    T = cfg.T
    output_image = cfg.output_image
    match_1_2 = np.argmin(D, axis=0)
    match_2_1 = np.argmin(D, axis=1)

    dist_2_1 = D[np.arange(D.shape[0]), match_2_1]

    indices = np.arange(D.shape[0])
    mask = (match_1_2[match_2_1] == indices) & (np.sqrt(dist_2_1) < T)

    ind1 = match_2_1[mask]
    ind2 = indices[mask]

    output_img = np.concatenate((image1, image2), axis=1)

    H, W = image1.shape[:2]

    for idx in ind1:
        x, y = pts1[idx]
        cv2.circle(output_img, (int(x), int(y)), cfg.radius, tuple(cfg.color_pur), -1)

    for idx in ind2:
        x, y = pts2[idx]
        cv2.circle(
            output_img, (int(x) + W, int(y)), cfg.radius, tuple(cfg.color_pur), -1
        )

    for i in range(len(ind1)):
        x1, y1 = pts1[ind1[i]]
        x2, y2 = pts2[ind2[i]]
        cv2.line(
            output_img,
            (int(x1), int(y1)),
            (int(x2) + W, int(y2)),
            tuple(cfg.color_gr),
            cfg.line_thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_image, output_img)


def main(cfg):
    image1 = cv2.imread(cfg.input_image_1)
    image2 = cv2.imread(cfg.input_image_2)
    D, pts1, pts2 = desc_distances(cfg, image1, image2)

    Cross_Check_Matching(cfg, D, image1, image2, pts1, pts2)


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("../params.yaml")[task]

    main(cfg)
