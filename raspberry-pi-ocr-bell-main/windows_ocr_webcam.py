import cv2
import numpy as np
import pytesseract
import argparse
from PIL import Image
import re
import time
from datetime import datetime
import os
import tempfile
import threading


def parse_args():
    p = argparse.ArgumentParser(description="Windows webcam OCR for red/pink clock digits")
    p.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    p.add_argument('--tesseract-path', type=str,
                   default=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                   help='Path to tesseract executable (Windows default)')
    p.add_argument('--image', type=str, default=None, help='Path to an image file to OCR instead of using webcam')
    p.add_argument('--min-area', type=int, default=800, help='Minimum contour area to consider')
    p.add_argument('--display', action='store_true', help='Show video window with annotations')
    p.add_argument('--show-preprocessed', action='store_true', help='Show preprocessed crop(s) used for OCR')
    p.add_argument('--debug-ocr', action='store_true', help='Print raw OCR outputs for each preprocessed region and whole-image attempts')
    p.add_argument('--resize-factor', type=float, default=1.0, help='Scale factor to resize frames for faster processing (0.5 = half size)')
    p.add_argument('--frame-skip', type=int, default=0, help='Skip processing for N frames between OCR passes (0 = process every frame)')
    p.add_argument('--save-on-detect', action='store_true', help='Save snapshot when time-like text detected')
    p.add_argument('--cooldown', type=float, default=3.0, help='Seconds to wait before repeating same detection')
    p.add_argument('--capture-interval', type=float, default=1.0, help='Seconds between camera snapshots to run OCR on')
    return p.parse_args()


def get_red_mask(hsv):
    # red range (two ranges in HSV) that also captures bright pink tones
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def report_detection(time_str, frame=None, save_on_detect=False, stamp=None):
    """Print and optionally overlay/save the detected time.

    If `frame` is provided, draw a large overlay and display it briefly.
    """
    if stamp is None:
        stamp = datetime.now().isoformat(sep=' ', timespec='seconds')
    print(f'Time detected: {time_str} at {stamp}')
    if save_on_detect and frame is not None:
        fn = f'detect_{time_str.replace(":", "-")}_{int(time.time())}.png'
        try:
            cv2.imwrite(fn, frame)
        except Exception:
            pass

    if frame is not None:
        disp = frame.copy()
        txt = f"{time_str}"
        # draw outline
        cv2.putText(disp, txt, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(disp, txt, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4, cv2.LINE_AA)
        try:
            cv2.imshow('Detected Time', disp)
            # show until keypress for image-mode; for webcam this will flash briefly
            if not hasattr(report_detection, '_webcam_mode') or not report_detection._webcam_mode:
                cv2.waitKey(0)
            else:
                cv2.waitKey(800)
        except Exception:
            pass


def preprocess_crop(crop):
    # Convert to gray and use adaptive thresholding to enhance digits
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # close small gaps and strengthen thin segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=1)
    return th


def robust_ocr(img):
    """Try several preprocessing variants and tesseract modes to extract time-like text."""
    configs = [
        '--psm 7 -c tessedit_char_whitelist=0123456789:',
        '--psm 6 -c tessedit_char_whitelist=0123456789:',
        '--psm 8 -c tessedit_char_whitelist=0123456789:'
    ]
    pattern = re.compile(r"\d{1,2}:\d{2}")

    # ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # limit tries to keep it reasonably fast
    for scale in (1.0, 1.5, 2.0):
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for invert in (False, True):
            im = 255 - resized if invert else resized
            # try a small set of morphology ops
            kernels = [None, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))]
            for k in kernels:
                proc = im.copy()
                if k is not None:
                    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, k)

                for cfg in configs:
                    try:
                        txt = pytesseract.image_to_string(proc, config=cfg)
                    except Exception:
                        txt = ''
                    s = txt.strip().replace(' ', '').replace('l', '1').replace('I', '1')
                    m = pattern.search(s)
                    if m:
                        return m.group(0)

    return None


def ocr_chars_from_image(bin_img):
    """Segment a binary preprocessed image into character blobs and OCR each with psm 10."""
    # Ensure binary image (0/255)
    img = bin_img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find internal contours (digits segments will be grouped into blobs)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 50:  # skip tiny noise
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return ''

    boxes = sorted(boxes, key=lambda b: b[0])
    digits = []
    for i, (x, y, w, h) in enumerate(boxes):
        ch = img[y:y+h, x:x+w]
        # pad and resize to give tesseract more pixels
        pad = 6
        ch = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ch = cv2.resize(ch, (64, 96), interpolation=cv2.INTER_CUBIC)
        try:
            out = pytesseract.image_to_string(ch, config='--psm 10 -c tessedit_char_whitelist=0123456789')
        except Exception:
            out = ''
        out = out.strip()
        if out:
            # keep only digits
            out = re.sub(r'[^0-9]', '', out)
        digits.append(out)

    return ''.join(digits)


def recognize_7seg(bin_img, debug=False):
    """Recognize a single seven-segment digit from a binary preprocessed image.

    Returns a single digit as string or None.
    """
    img = bin_img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize to a canonical size
    h, w = img.shape
    if h == 0 or w == 0:
        return None
    target_h = 100
    target_w = int(w * (target_h / h))
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Define segment ROIs relative to size
    th = 0.25  # fraction of white pixels to consider segment ON (lowered to detect thin segments)
    H, W = img.shape
    segs = []
    # a: top horizontal
    segs.append((int(W*0.20), int(H*0.05), int(W*0.60), int(H*0.15)))
    # b: upper-right vertical (wider ROI to handle narrow digits)
    segs.append((int(W*0.70), int(H*0.10), int(W*0.25), int(H*0.40)))
    # c: lower-right vertical (wider ROI)
    segs.append((int(W*0.70), int(H*0.50), int(W*0.25), int(H*0.40)))
    # d: bottom horizontal
    segs.append((int(W*0.20), int(H*0.85), int(W*0.60), int(H*0.10)))
    # e: lower-left vertical (wider ROI)
    segs.append((int(W*0.05), int(H*0.50), int(W*0.25), int(H*0.40)))
    # f: upper-left vertical (wider ROI)
    segs.append((int(W*0.05), int(H*0.10), int(W*0.25), int(H*0.40)))
    # g: middle horizontal
    segs.append((int(W*0.20), int(H*0.45), int(W*0.60), int(H*0.10)))

    on = []
    for (x, y, ww, hh) in segs:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + ww)
        y1 = min(H, y + hh)
        if x1 <= x0 or y1 <= y0:
            on.append(False)
            continue
        roi = img[y0:y1, x0:x1]
        # compute fraction of white pixels
        frac = (roi > 128).sum() / (roi.size + 1e-9)
        on.append(frac >= th)
        if debug:
            print(f'seg region {(x0,y0,x1,y1)} frac={frac:.2f} on={frac>=th}')

    # map segments [a,b,c,d,e,f,g] to digit
    mask = 0
    for i, val in enumerate(on):
        if val:
            mask |= (1 << i)

    seg_to_digit = {
        0b0111111: '0',
        0b0000110: '1',
        0b1011011: '2',
        0b1001111: '3',
        0b1100110: '4',
        0b1101101: '5',
        0b1111101: '6',
        0b0000111: '7',
        0b1111111: '8',
        0b1101111: '9'
    }

    # Our bit order is [a,b,c,d,e,f,g] -> compute expected masks accordingly
    # The above mapping uses same bit order
    digit = seg_to_digit.get(mask)
    if debug:
        print(f'7seg mask={bin(mask)} digit={digit}')
    # Fallback: if no mapping found, try narrow/tall vertical-stroke detection for '1'
    if digit is None:
        try:
            # aspect ratio: narrow when width << height
            aspect = W / float(H)
            if aspect < 0.5:
                # column-wise fraction of white pixels
                col_sums = (img > 128).sum(axis=0).astype(float) / (H + 1e-9)
                max_frac = float(col_sums.max())
                left_frac = float(col_sums[:W//2].mean()) if W//2 > 0 else 0.0
                right_frac = float(col_sums[W//2:].mean()) if W - W//2 > 0 else 0.0
                # if a vertical stroke exists, it's likely a '1'
                # For very narrow digits, accept a centered vertical stroke as '1'.
                if max_frac > 0.08:
                    digit = '1'
                    if debug:
                        idx = int(np.argmax(col_sums))
                        print(f'7seg fallback vertical stroke max_frac={max_frac:.2f} idx={idx} left_frac={left_frac:.2f} right_frac={right_frac:.2f} -> 1 (centered/narrow)')
        except Exception:
            pass
    return digit


def infer_time_from_text(raw):
    """Try to infer H:MM from a raw OCR string that may lack a colon.

    Examples:
      '1200' -> '12:00'
      '900'  -> '9:00'
      '12 00'-> '12:00'
    Returns a string like 'H:MM' or None.
    """
    if not raw:
        return None
    s = re.sub(r'[^0-9]', '', raw)
    if len(s) == 4:
        return f"{int(s[:2])}:{s[2:]}"
    if len(s) == 3:
        return f"{int(s[:1])}:{s[1:]}"
    if len(s) == 2:
        # ambiguous: treat as hour only if between 0-23 and append :00
        h = int(s)
        if 0 <= h <= 23:
            return f"{h}:00"
        return None
    return None


def run(camera_index, tesseract_path, min_area, display, save_on_detect, cooldown):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Note: `args` isn't directly available here; we'll read runtime-controlled
    # parameters from environment by expecting caller to pass appropriately.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Error: cannot open camera index', camera_index)
        return

    last_seen = {}  # text -> timestamp
    pattern = re.compile(r"\d{1,2}:\d{2}")

    # Grab processing tuning options from globals set by caller (parse_args)
    resize_factor = getattr(run, 'resize_factor', 1.0)
    frame_skip = getattr(run, 'frame_skip', 0)
    show_pre = getattr(run, 'show_preprocessed', False)

    try:
        frame_idx = 0
        last_capture = 0.0
        capture_interval = getattr(run, 'capture_interval', 1.0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame')
                break

            frame_idx += 1
            # Optionally skip frames to reduce CPU usage and apparent lag
            if frame_skip and (frame_idx % (frame_skip + 1)) != 1:
                if display:
                    cv2.imshow('Red-text OCR', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # Resize for faster processing; keep the original for display
            proc_frame = frame
            if resize_factor != 1.0 and resize_factor > 0:
                proc_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

            hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            mask = get_red_mask(hsv)

            # clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Collect crops to try combined OCR (helps when digits are separate blobs)
            crops = []
            coords = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                # include narrow/tall contours (likely '1') even if area is small
                is_narrow = (h > 3 * w and w > 2 and area > 20)
                if area < min_area and not is_narrow:
                    continue
                pad = 8
                # scale coordinates back to original frame if we resized
                if proc_frame is not frame:
                    scale = 1.0 / resize_factor
                    x0 = int(max(0, (x - pad) * scale))
                    y0 = int(max(0, (y - pad) * scale))
                    x1 = int(min(frame.shape[1], (x + w + pad) * scale))
                    y1 = int(min(frame.shape[0], (y + h + pad) * scale))
                else:
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(frame.shape[1], x + w + pad)
                    y1 = min(frame.shape[0], y + h + pad)

                crop = frame[y0:y1, x0:x1]
                proc = preprocess_crop(crop)
                crops.append(proc)
                coords.append((x0, y0, x1, y1))

                if show_pre:
                    cv2.imshow('Preprocessed', proc)

                # draw rectangles for display
                if display:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Try OCR on combined crops (left-to-right) for better chance at matching time
            if crops:
                # sort by x coordinate
                pairs = sorted(zip(coords, crops), key=lambda pc: pc[0][0])
                sorted_crops = [pc[1] for pc in pairs]
                # add small separators between crops
                sep = 10
                heights = [c.shape[0] for c in sorted_crops]
                max_h = max(heights)
                norm_crops = [cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for c in sorted_crops]
                combined = norm_crops[0]
                for c in norm_crops[1:]:
                    combined = np.hstack((combined, np.zeros((max_h, sep), dtype=combined.dtype), c))

                combined_found = robust_ocr(combined)
                if combined_found:
                    now = time.time()
                    stamp = datetime.now().isoformat(sep=' ', timespec='seconds')
                    report_detection(combined_found, frame=frame, save_on_detect=save_on_detect, stamp=stamp)
                else:
                    # Try char-seg OCR on combined image
                    chars = ocr_chars_from_image(combined)
                    if chars:
                        inferred = infer_time_from_text(chars)
                        if inferred:
                            now = time.time()
                            stamp = datetime.now().isoformat(sep=' ', timespec='seconds')
                            report_detection(inferred, frame=frame, save_on_detect=save_on_detect, stamp=stamp)
                            continue

                    # try seven-seg per crop
                    seg_digits = []
                    for c in sorted_crops:
                        d = recognize_7seg(c)
                        seg_digits.append(d if d else '')
                    joined = ''.join(seg_digits)
                    if joined:
                        inferred = infer_time_from_text(joined)
                        if inferred:
                            now = time.time()
                            stamp = datetime.now().isoformat(sep=' ', timespec='seconds')
                            report_detection(inferred, frame=frame, save_on_detect=save_on_detect, stamp=stamp)
                            continue

                    # if that fails, try raw tesseract without whitelist and attempt inference
                    raw_comb = pytesseract.image_to_string(combined, config='--psm 7')
                    if raw_comb:
                        inferred = infer_time_from_text(raw_comb)
                        if inferred:
                            now = time.time()
                            stamp = datetime.now().isoformat(sep=' ', timespec='seconds')
                            print(f'Inferred {inferred} from combined raw "{raw_comb.strip()}" at {stamp}')
                            if save_on_detect:
                                cv2.imwrite(f'detect_{inferred.replace(":", "-")}_{int(now)}.png', frame)

                if display:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if display:
                cv2.imshow('Red-text OCR', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Periodically save a snapshot and run OCR on it in a background thread
            now = time.time()
            if capture_interval and (now - last_capture) >= float(capture_interval):
                last_capture = now
                # write to system temp directory
                try:
                    tmpdir = tempfile.gettempdir()
                    tmpfn = os.path.join(tmpdir, f'camera_snap_{int(now)}.png')
                    cv2.imwrite(tmpfn, frame)
                    # run OCR on the saved photo in background to avoid blocking
                    dbg = getattr(run, 'debug_ocr', False)
                    t = threading.Thread(target=process_image_file, args=(tmpfn, tesseract_path, min_area, save_on_detect, dbg), daemon=True)
                    t.start()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def process_image_file(image_path, tesseract_path, min_area, save_on_detect, debug_ocr=False):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    img = cv2.imread(image_path)
    if img is None:
        print('Error: cannot read image', image_path)
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_red_mask(hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_any = False
    pattern = re.compile(r"\d{1,2}:\d{2}")

    idx = 0
    crops = []
    coords = []
    region_results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        is_narrow = (h > 3 * w and w > 2 and area > 20)
        if area < min_area and not is_narrow:
            continue
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad)
        y1 = min(img.shape[0], y + h + pad)
        crop = img[y0:y1, x0:x1]
        proc = preprocess_crop(crop)
        # Save and show preprocessed crop for debugging
        pre_fn = f'preprocessed_region_{idx}.png'
        cv2.imwrite(pre_fn, proc)
        try:
            cv2.imshow('Preprocessed', proc)
            cv2.waitKey(200)
        except Exception:
            pass
        if debug_ocr:
            raw = pytesseract.image_to_string(proc, config='--psm 7 -c tessedit_char_whitelist=0123456789:')
            print(f'Region {idx} raw OCR (whitelist psm7): "{raw.strip()}"')
            # additional debug passes: try without whitelist, different psm, and inverted image
            try:
                pil = Image.open(pre_fn).convert('L')
            except Exception:
                pil = None
            extra_cfgs = ['','--psm 7','--psm 6','--psm 8']
            for cfg in extra_cfgs:
                try:
                    if pil is not None:
                        out = pytesseract.image_to_string(pil, config=cfg)
                    else:
                        out = pytesseract.image_to_string(proc, config=cfg)
                except Exception as e:
                    out = f'ERROR: {e}'
                print(f'  Region {idx} raw OCR cfg="{cfg}": "{str(out).strip()}"')
            # inverted
            try:
                if pil is not None:
                    inv = Image.fromarray(255 - np.array(pil))
                    out = pytesseract.image_to_string(inv, config='--psm 7')
                    print(f'  Region {idx} raw OCR inverted psm7: "{out.strip()}"')
            except Exception as e:
                print(f'  Region {idx} inverted OCR ERROR: {e}')
        idx += 1
        # keep lists for combined-crops fallback
        crops.append(proc)
        coords.append((x0, y0, x1, y1))

        # Try robust OCR on this region
        found_text = robust_ocr(proc)
        # if robust OCR didn't find H:MM, try char-level segmentation OCR
        if not found_text:
            chars = ocr_chars_from_image(proc)
            if chars:
                # try to infer time from collected digits
                inferred = infer_time_from_text(chars)
                if inferred:
                    found_text = inferred
                    if debug_ocr:
                        print(f'Region {idx} char-level OCR digits: "{chars}" -> inferred "{inferred}"')
                else:
                    if debug_ocr:
                        print(f'Region {idx} char-level OCR digits (no inference): "{chars}"')
            # try seven-seg recognizer per-region
            if not found_text:
                segd = recognize_7seg(proc, debug=debug_ocr)
                if segd:
                    # single digit recognized
                    found_text = segd
                    if debug_ocr:
                        print(f'Region {idx} 7-seg digit: "{segd}"')
        if debug_ocr and not found_text:
            # also show what robust_ocr tried (best-effort): run a simple pass and show
            try_text = pytesseract.image_to_string(proc, config='--psm 6 -c tessedit_char_whitelist=0123456789:')
            print(f'Region {idx} robust fallback raw: "{try_text.strip()}"')
            # attempt to infer a time from the raw region text
            inferred_region = infer_time_from_text(try_text)
            if inferred_region:
                found_text = inferred_region
        if found_text:
            # collect region-level results and coordinates; report after processing all regions
            found_any = True
            region_results.append((x0, found_text, (x0, y0, x1, y1)))

    # After processing all regions, try to assemble a final time from collected region results
    if region_results:
        # Prefer any region that already contains H:MM
        time_re = re.compile(r"\d{1,2}:\d{2}")
        reported = False
        for _, txt, rect in region_results:
            if time_re.search(txt):
                report_detection(txt, frame=img, save_on_detect=save_on_detect)
                reported = True
                break
        if not reported:
            # assemble single-digit results left-to-right
            region_results.sort(key=lambda r: r[0])
            digits = ''.join([r[1] for r in region_results if r[1] is not None])
            inferred = infer_time_from_text(digits)
            if inferred:
                report_detection(inferred, frame=img, save_on_detect=save_on_detect)
                reported = True
        if reported:
            found_any = True

    # Combined-crops fallback: if no region produced a time, try concatenating regions left-to-right
    if not found_any and crops:
        pairs = sorted(zip(coords, crops), key=lambda pc: pc[0][0])
        sorted_crops = [pc[1] for pc in pairs]
        sep = 10
        heights = [c.shape[0] for c in sorted_crops]
        max_h = max(heights)
        norm_crops = [cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for c in sorted_crops]
        combined = norm_crops[0]
        for c in norm_crops[1:]:
            combined = np.hstack((combined, np.zeros((max_h, sep), dtype=combined.dtype), c))

        if debug_ocr:
            cv2.imwrite('debug_combined.png', combined)

        combined_found = robust_ocr(combined)
        if combined_found:
            report_detection(combined_found, frame=img, save_on_detect=save_on_detect)
            found_any = True
        else:
            chars = ocr_chars_from_image(combined)
            if chars:
                inferred = infer_time_from_text(chars)
                if inferred:
                    print('Inferred', inferred, 'from combined char-level')
                    if save_on_detect:
                        cv2.imwrite(f'detect_{inferred.replace(":", "-")}_combined.png', img)
                    found_any = True
            if not found_any:
                seg_digits = [recognize_7seg(c) or '' for c in sorted_crops]
                joined = ''.join(seg_digits)
                if joined:
                    inferred = infer_time_from_text(joined)
                    if inferred:
                        print('Inferred', inferred, 'from combined 7-seg', joined)
                        if save_on_detect:
                            cv2.imwrite(f'detect_{inferred.replace(":", "-")}_combined.png', img)
                        found_any = True

    # fallback: if no red contours gave results, run OCR on the whole preprocessed image
    if not found_any:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        h, w = thresh.shape
        thresh = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        config = '--psm 6 -c tessedit_char_whitelist=0123456789:'
        # Try robust OCR on the whole preprocessed image too
        if debug_ocr:
            raw_whole = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789:')
            print(f'Whole-image raw OCR: "{raw_whole.strip()}"')
        whole_found = robust_ocr(thresh)
        if whole_found:
            print('Detected', whole_found, 'from whole-image OCR')
            if save_on_detect:
                cv2.imwrite(f'detect_{whole_found.replace(":", "-")}_whole.png', img)
        else:
            # try to infer from raw whole-image OCR
            if debug_ocr:
                inferred_whole = infer_time_from_text(raw_whole)
            else:
                # get raw_whole now if not debug
                raw_whole = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789:')
                inferred_whole = infer_time_from_text(raw_whole)

            if inferred_whole:
                print(f'Inferred {inferred_whole} from whole-image raw "{raw_whole.strip()}"')
                if save_on_detect:
                    cv2.imwrite(f'detect_{inferred_whole.replace(":", "-")}_whole.png', img)
            else:
                print('No time-like text found in image')


if __name__ == '__main__':
    args = parse_args()
    print('Using tesseract at', args.tesseract_path)
    if args.image:
        process_image_file(args.image, args.tesseract_path, args.min_area, args.save_on_detect, debug_ocr=args.debug_ocr)
    else:
        # attach runtime tuning parameters to the run function so it can read them
        run.resize_factor = args.resize_factor
        run.frame_skip = args.frame_skip
        run.show_preprocessed = args.show_preprocessed
        run.debug_ocr = args.debug_ocr
        run.capture_interval = args.capture_interval
        # tell report_detection we're in webcam mode so overlays auto-dismiss
        try:
            report_detection._webcam_mode = True
        except Exception:
            pass
        run(args.camera, args.tesseract_path, args.min_area, args.display, args.save_on_detect, args.cooldown)
