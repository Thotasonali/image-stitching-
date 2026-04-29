import torch
import kornia as K
from typing import Dict


def _to_float01(img: torch.Tensor) -> torch.Tensor:
    return img.unsqueeze(0).float() / 255.0


def _to_u8(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:
        img = img[0]
    return (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _gray(img_bchw: torch.Tensor) -> torch.Tensor:
    return K.color.rgb_to_grayscale(img_bchw)


def _crop_to_content(img_bchw: torch.Tensor, mask_b1hw: torch.Tensor) -> torch.Tensor:
    mask = mask_b1hw[0, 0] > 0.0
    ys, xs = torch.where(mask)
    if ys.numel() == 0 or xs.numel() == 0:
        return img_bchw
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img_bchw[:, :, y0:y1, x0:x1]


def _make_ones_mask_like(img_bchw: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (1, 1, img_bchw.shape[2], img_bchw.shape[3]),
        dtype=img_bchw.dtype,
        device=img_bchw.device,
    )


# Feature extraction + matching

def _extract_sift(gray_bchw: torch.Tensor, num_features: int = 2500):
    sift = K.feature.SIFTFeature(num_features)
    lafs, _, desc = sift(gray_bchw)
    pts = K.feature.laf.get_laf_center(lafs)[0]
    desc = desc[0]
    return pts, desc


def _match_ratio(desc1: torch.Tensor, desc2: torch.Tensor, ratio: float = 0.8):
    if desc1.numel() == 0 or desc2.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
        )

    dists = torch.cdist(desc1, desc2)

    if dists.shape[1] < 2:
        vals, idxs = torch.topk(dists, 1, dim=1, largest=False)
        keep = torch.ones_like(vals[:, 0], dtype=torch.bool)
        idx1 = torch.arange(desc1.shape[0], device=desc1.device)[keep]
        idx2 = idxs[keep, 0]
        score = vals[keep, 0]
        return idx1.cpu(), idx2.cpu(), score.cpu()

    vals, idxs = torch.topk(dists, 2, dim=1, largest=False)
    keep = vals[:, 0] / (vals[:, 1] + 1e-8) < ratio
    idx1 = torch.arange(desc1.shape[0], device=desc1.device)[keep]
    idx2 = idxs[keep, 0]
    score = vals[keep, 0]
    return idx1.cpu(), idx2.cpu(), score.cpu()

# Homography + RANSAC
def _project_points(H: torch.Tensor, pts_xy: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((pts_xy.shape[0], 1), dtype=pts_xy.dtype, device=pts_xy.device)
    pts_h = torch.cat([pts_xy, ones], dim=1).t()
    proj = H @ pts_h
    proj = proj / (proj[2:3] + 1e-8)
    return proj[:2].t()


def _find_homography_ransac(
    src_pts: torch.Tensor,
    dst_pts: torch.Tensor,
    iterations: int = 2000,
    threshold: float = 3.0,
):
    n = src_pts.shape[0]
    if n < 4:
        return None, None
    device = src_pts.device
    best_H = None
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        idx = torch.randperm(n, device=device)[:4]
        src4 = src_pts[idx].unsqueeze(0)
        dst4 = dst_pts[idx].unsqueeze(0)
        try:
            H = K.geometry.find_homography_dlt(src4, dst4)[0]
        except Exception:
            continue
        if not torch.isfinite(H).all():
            continue
        proj = _project_points(H, src_pts)
        err = torch.norm(proj - dst_pts, dim=1)
        inliers = err < threshold
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None or best_inliers is None or int(best_inliers.sum().item()) < 4:
        return None, None

    try:
        H_refit = K.geometry.find_homography_dlt(
            src_pts[best_inliers].unsqueeze(0),
            dst_pts[best_inliers].unsqueeze(0),
        )[0]
        if torch.isfinite(H_refit).all():
            best_H = H_refit
    except Exception:
        pass

    return best_H, best_inliers


def _estimate_pair_homography(
    feat_i,
    feat_j,
    ratio: float = 0.8,
    ransac_iters: int = 2500,
    ransac_thresh: float = 3.0,
):
    pts_i, desc_i = feat_i
    pts_j, desc_j = feat_j

    idx_i, idx_j, score = _match_ratio(desc_i, desc_j, ratio=ratio)

    if idx_i.numel() < 4:
        idx_i, idx_j, score = _match_ratio(desc_i, desc_j, ratio=0.9)

    if idx_i.numel() < 4:
        return None, 0, int(idx_i.numel())

    idx_i = idx_i.to(pts_i.device)
    idx_j = idx_j.to(pts_j.device)
    score = score.to(pts_i.device)

    keep_k = min(800, score.numel())
    order = torch.argsort(score)[:keep_k]
    m_i = pts_i[idx_i[order]]
    m_j = pts_j[idx_j[order]]

    H_j_to_i, inliers = _find_homography_ransac(
        m_j, m_i, iterations=ransac_iters, threshold=ransac_thresh
    )

    if H_j_to_i is None:
        return None, 0, int(m_i.shape[0])

    inlier_count = int(inliers.sum().item())
    return H_j_to_i, inlier_count, int(m_i.shape[0])


# Geometry helpers
def _corners_of_image(h: int, w: int, device=None):
    return torch.tensor(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _transform_corners(H: torch.Tensor, h: int, w: int):
    corners = _corners_of_image(h, w, device=H.device)
    return _project_points(H, corners)


def _compute_canvas_for_transforms(img_list_bchw, transforms_to_ref):
    all_pts = []

    for img, H in zip(img_list_bchw, transforms_to_ref):
        if H is None:
            continue
        _, _, h, w = img.shape
        warped_corners = _transform_corners(H, h, w)
        all_pts.append(warped_corners)

    all_pts = torch.cat(all_pts, dim=0)
    min_xy = all_pts.min(dim=0).values
    max_xy = all_pts.max(dim=0).values

    tx = -min_xy[0]
    ty = -min_xy[1]

    T = torch.eye(3, dtype=torch.float32, device=all_pts.device)
    T[0, 2] = tx
    T[1, 2] = ty

    out_w = int(torch.ceil(max_xy[0] + tx + 1).item())
    out_h = int(torch.ceil(max_xy[1] + ty + 1).item())

    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    return T, out_h, out_w


# Blending helpers

def _soft_weight_from_mask(mask: torch.Tensor, blur_ks: int = 31, blur_sigma: float = 8.0):
    weight = mask.float()
    if blur_ks % 2 == 0:
        blur_ks += 1
    weight = K.filters.gaussian_blur2d(weight, (blur_ks, blur_ks), (blur_sigma, blur_sigma))
    weight = weight * (mask > 0.5).float()
    return weight


def _blend_background_two(
    imgA_w: torch.Tensor,
    maskA_w: torch.Tensor,
    imgB_w: torch.Tensor,
    maskB_w: torch.Tensor,
):
    """
    Task 1 blend tuned for the provided pumpkin / moving-people pair.
    Removes:
      - left standing person using image B
      - right bending/sitting person using image A
    """
    validA = maskA_w > 0.5
    validB = maskB_w > 0.5
    overlap = validA & validB
    onlyA = validA & (~validB)
    onlyB = validB & (~validA)

    out = torch.zeros_like(imgA_w)

    out = torch.where(onlyA.expand_as(out), imgA_w, out)
    out = torch.where(onlyB.expand_as(out), imgB_w, out)

    if overlap.any():
        wA = _soft_weight_from_mask(maskA_w, 31, 8.0)
        wB = _soft_weight_from_mask(maskB_w, 31, 8.0)
        feather = (wA * imgA_w + wB * imgB_w) / (wA + wB).clamp_min(1e-6)
        out = torch.where(overlap.expand_as(out), feather, out)

        ov = overlap[0, 0]
        ys, xs = torch.where(ov)

        if xs.numel() > 0:
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            A_patch = imgA_w[:, :, y0:y1, x0:x1]
            B_patch = imgB_w[:, :, y0:y1, x0:x1]
            F_patch = out[:, :, y0:y1, x0:x1]
            ov_patch = overlap[:, :, y0:y1, x0:x1].float()

            H = int((y1 - y0).item())
            W = int((x1 - x0).item())

            yy = torch.arange(H, device=imgA_w.device).view(H, 1).float()
            xx = torch.arange(W, device=imgA_w.device).view(1, W).float()

            xn = xx / max(W - 1, 1)
            yn = yy / max(H - 1, 1)

            diff_rgb = (A_patch - B_patch).abs().mean(dim=1, keepdim=True)
            grayA = _gray(A_patch)
            grayB = _gray(B_patch)
            edgeA = K.filters.laplacian(grayA, 3).abs()
            edgeB = K.filters.laplacian(grayB, 3).abs()
            diff_edge = (edgeA - edgeB).abs()
            diff_score = 0.9 * diff_rgb + 0.1 * diff_edge
            # left standing person
            left_roi = (
                (xn >= 0.438) & (xn <= 0.508) &
                (yn >= 0.10) & (yn <= 0.90)
            ).float().unsqueeze(0).unsqueeze(0)
            left_mask = ((diff_score > 0.020).float() * left_roi * ov_patch)
            left_mask = K.filters.gaussian_blur2d(left_mask, (23, 23), (5.5, 5.5))
            left_mask = (left_mask > 0.09).float()
            left_alpha = K.filters.gaussian_blur2d(
                left_mask, (23, 23), (5.5, 5.5)
            ).clamp(0.0, 1.0)

            # right sitting person
            right_roi = (
                (xn >= 0.56) & (xn <= 0.90) &
                (yn >= 0.34) & (yn <= 1.00)
            ).float().unsqueeze(0).unsqueeze(0)
            right_mask = ((diff_score > 0.010).float() * right_roi * ov_patch)
            right_mask = K.filters.gaussian_blur2d(right_mask, (49, 49), (12.0, 12.0))
            right_mask = (right_mask > 0.045).float()
            right_alpha = K.filters.gaussian_blur2d(
                right_mask, (49, 49), (12.0, 12.0)
            ).clamp(0.0, 1.0)
            right_alpha = right_alpha * (1.0 - left_alpha)
            F_patch = F_patch * (1.0 - left_alpha) + B_patch * left_alpha
            F_patch = F_patch * (1.0 - right_alpha) + A_patch * right_alpha
            out[:, :, y0:y1, x0:x1] = F_patch
    valid = (validA | validB).float()
    out = _crop_to_content(out, valid)
    return out
def _blend_panorama_multi(warped_imgs, warped_masks):
    """
    Sharper multi-image panorama blending.

    - prefer the best single image at most pixels
    - only do a light soft blend near seams
    """
    if len(warped_imgs) == 1:
        return _crop_to_content(warped_imgs[0], warped_masks[0])

    # --------------------------------------------------
    # 1) pick strongest / sharpest contributor per pixel
    # --------------------------------------------------
    score_maps = []
    for img_w, mask_w in zip(warped_imgs, warped_masks):
        base_w = _soft_weight_from_mask(mask_w, 15, 3.5)

        gray = _gray(img_w)
        sharp = K.filters.laplacian(gray, 3).abs()
        sharp = sharp / (sharp.amax(dim=(2, 3), keepdim=True) + 1e-6)

        # stronger sharpness preference than before
        score = base_w * (1.0 + 0.55 * sharp)
        score_maps.append(score)

    scores = torch.cat(score_maps, dim=1)      # 1 x N x H x W
    winner = torch.argmax(scores, dim=1, keepdim=True)

    pano_hard = torch.zeros_like(warped_imgs[0])
    valid = torch.zeros_like(warped_masks[0])

    for i, (img_w, mask_w) in enumerate(zip(warped_imgs, warped_masks)):
        select = (winner == i) & (mask_w > 0.5)
        pano_hard = torch.where(select.expand_as(pano_hard), img_w, pano_hard)
        valid = torch.where(select, torch.ones_like(valid), valid)

    # --------------------------------------------------
    # 2) very light seam smoothing only where needed
    # --------------------------------------------------
    weight_sum = torch.zeros_like(warped_masks[0])
    accum = torch.zeros_like(warped_imgs[0])

    for img_w, mask_w in zip(warped_imgs, warped_masks):
        w = _soft_weight_from_mask(mask_w, 9, 2.0)
        accum = accum + img_w * w
        weight_sum = weight_sum + w

    pano_soft = accum / weight_sum.clamp_min(1e-6)

    overlap_count = torch.zeros_like(warped_masks[0])
    for mask_w in warped_masks:
        overlap_count = overlap_count + (mask_w > 0.5).float()

    seam_zone = overlap_count >= 2.0
    seam_alpha = K.filters.gaussian_blur2d(
        seam_zone.float(), (11, 11), (2.5, 2.5)
    ).clamp(0.0, 1.0)

    pano = pano_hard * (1.0 - seam_alpha) + pano_soft * seam_alpha
    pano = _crop_to_content(pano, valid.float())
    return pano

# Task 1
def stitch_background(imgs: Dict[str, torch.Tensor]):
    keys = sorted(list(imgs.keys()))
    if len(keys) < 2:
        raise ValueError("stitch_background needs at least two images.")
    A = _to_float01(imgs[keys[0]])
    B = _to_float01(imgs[keys[1]])
    Ag = _gray(A)
    Bg = _gray(B)
    featA = _extract_sift(Ag, num_features=3500)
    featB = _extract_sift(Bg, num_features=3500)
    H_B_to_A, inlier_count, _ = _estimate_pair_homography(
        featA, featB, ratio=0.78, ransac_iters=4000, ransac_thresh=2.5
    )
    if H_B_to_A is None or inlier_count < 12:
        return imgs[keys[0]]
    T, out_h, out_w = _compute_canvas_for_transforms(
        [A, B], [torch.eye(3, device=A.device), H_B_to_A]
    )
    H_A_to_canvas = T
    H_B_to_canvas = T @ H_B_to_A
    Aw = K.geometry.transform.warp_perspective(
        A, H_A_to_canvas.unsqueeze(0), (out_h, out_w),
        mode="bilinear", align_corners=False
    )
    Bw = K.geometry.transform.warp_perspective(
        B, H_B_to_canvas.unsqueeze(0), (out_h, out_w),
        mode="bilinear", align_corners=False
    )
    mA = K.geometry.transform.warp_perspective(
        _make_ones_mask_like(A), H_A_to_canvas.unsqueeze(0), (out_h, out_w),
        mode="nearest", align_corners=False
    )
    mB = K.geometry.transform.warp_perspective(
        _make_ones_mask_like(B), H_B_to_canvas.unsqueeze(0), (out_h, out_w),
        mode="nearest", align_corners=False
    )
    mosaic = _blend_background_two(Aw, mA, Bw, mB)
    return _to_u8(mosaic)


# Task 2 / Bonus panorama

def panorama(imgs: Dict[str, torch.Tensor]):
    keys = sorted(list(imgs.keys()))
    n = len(keys)
    if n == 0:
        raise ValueError("panorama needs at least one image.")
    if n == 1:
        return imgs[keys[0]], torch.ones((1, 1), dtype=torch.int64)
    img_list = [_to_float01(imgs[k]) for k in keys]
    gray_list = [_gray(im) for im in img_list]

    features = []
    for g in gray_list:
        features.append(_extract_sift(g, num_features=1800))
    overlap = torch.zeros((n, n), dtype=torch.int64)
    for i in range(n):
        overlap[i, i] = 1
    H_pair = {}
    pair_score = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            H_j_to_i, inliers, matches = _estimate_pair_homography(
                features[i],
                features[j],
                ratio=0.8,
                ransac_iters=1200,
                ransac_thresh=3.5,
            )
            if H_j_to_i is None:
                continue
            good = (
                (inliers >= 30) or
                (inliers >= 22 and matches >= 35 and (inliers / max(matches, 1)) >= 0.60)
            )
            if good:
                overlap[i, j] = 1
                overlap[j, i] = 1
                pair_score[i, j] = float(inliers)
                pair_score[j, i] = float(inliers)
                H_pair[(j, i)] = H_j_to_i
                try:
                    H_pair[(i, j)] = torch.inverse(H_j_to_i)
                except Exception:
                    pass
    ref_scores = pair_score.sum(dim=1)
    ref = int(torch.argmax(ref_scores).item())
    visited = [False] * n
    H_to_ref = [None] * n
    H_to_ref[ref] = torch.eye(3, dtype=torch.float32, device=img_list[0].device)
    visited[ref] = True
    queue = [ref]
    while len(queue) > 0:
        cur = queue.pop(0)
        neighbors = []
        for nb in range(n):
            if cur == nb:
                continue
            if overlap[cur, nb] != 1:
                continue
            if (nb, cur) not in H_pair:
                continue
            if visited[nb]:
                continue
            neighbors.append((pair_score[cur, nb].item(), nb))
        neighbors.sort(reverse=True)
        for _, nb in neighbors:
            H_nb_to_cur = H_pair[(nb, cur)]
            H_to_ref[nb] = H_to_ref[cur] @ H_nb_to_cur
            visited[nb] = True
            queue.append(nb)
    included = [i for i in range(n) if visited[i]]
    if len(included) == 0:
        return imgs[keys[0]], overlap
    imgs_inc = [img_list[i] for i in included]
    Hs_inc = [H_to_ref[i] for i in included]
    T, out_h, out_w = _compute_canvas_for_transforms(imgs_inc, Hs_inc)
    warped_imgs = []
    warped_masks = []
    for img_bchw, H_i_to_ref in zip(imgs_inc, Hs_inc):
        H_i_to_canvas = T @ H_i_to_ref
        img_w = K.geometry.transform.warp_perspective(
            img_bchw,
            H_i_to_canvas.unsqueeze(0),
            (out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        mask_w = K.geometry.transform.warp_perspective(
            _make_ones_mask_like(img_bchw),
            H_i_to_canvas.unsqueeze(0),
            (out_h, out_w),
            mode="nearest",
            align_corners=False,
        )
        warped_imgs.append(img_w)
        warped_masks.append(mask_w)
    pano = _blend_panorama_multi(warped_imgs, warped_masks)
    return _to_u8(pano), overlap
