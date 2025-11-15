'''import argparse
import time
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to best.onnx')
    p.add_argument('--imgsz', type=int, default=224, help='Input size (square)')
    p.add_argument('--camera', type=int, default=0, help='Webcam index (0=default)')
    p.add_argument('--labels', type=str, default="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,1,2,3,4,5,6,7,8,9")
    p.add_argument('--smooth', type=int, default=7, help='Temporal smoothing window')
    p.add_argument('--conf_thresh', type=float, default=0.50, help='Min confidence to display')
    p.add_argument("--mirror", action="store_true", help="Flip camera horizontally")

    return p.parse_args()

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def preprocess(frame, size):
    # Center-crop to square then resize to (size, size)
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0:y0+side, x0:x0+side]
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    # BGR->RGB, to float32, scale 0..1, NCHW
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)        # NCHW
    return img

def main():
    args = parse_args()
    labels = args.labels.split(',')
    num_classes = len(labels)

    # ONNX Runtime session
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print('Error: cannot open camera')
        return

    smooth_q = deque(maxlen=max(args.smooth, 1))
    t_last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

       

        inp = preprocess(frame, args.imgsz)
        logits = sess.run([out_name], {in_name: inp})[0].squeeze()  # shape (num_classes,)
        probs = softmax(logits)
        top_idx = int(np.argmax(probs))
        top_p = float(probs[top_idx])
        smooth_q.append((top_idx, top_p))

        # Temporal smoothing by majority vote on indices, then avg prob
        idxs = [i for i, _ in smooth_q]
        top_sm = max(set(idxs), key=idxs.count)
        avg_p = float(np.mean([p for i, p in smooth_q if i == top_sm]))

        # FPS
        now = time.time()
        dt = now - t_last
        t_last = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        # Overlay
        label = labels[top_sm] if 0 <= top_sm < num_classes else '?'
        text = f'{label}  {avg_p:.2f}  {fps:.1f} FPS'
        if avg_p >= args.conf_thresh:
            color = (40, 200, 40)
        else:
            color = (60, 60, 200)
        cv2.rectangle(frame, (8, 8), (320, 48), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        cv2.imshow('ONNX Runtime Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()'''
import json, os
STAT_PATH = os.path.join(os.path.dirname(__file__), 'latest_stats.json')
_last_write = 0.0
def write_stats(label, conf, fps, hz=4.0):
    global _last_write
    now = time.time()
    if now - _last_write >= 1.0 / hz:
        try:
            with open(STAT_PATH, 'w') as f:
                json.dump({"label": label, "conf": round(float(conf), 3), "fps": round(float(fps), 1)}, f)
        except Exception:
            pass
        _last_write = now

import argparse
import time
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to best.onnx')
    p.add_argument('--imgsz', type=int, default=224, help='Input size (square)')
    p.add_argument('--camera', type=int, default=0, help='Webcam index (0=default)')
    # Default order matches your folder names: 1..9, then A..Z
    p.add_argument(
        '--labels',
        type=str,
        default="Z,Y,X,W,V,U,T,S,R,Q,P,O,N,M,L,K,J,I,H,G,F,E,1,D,C,B,A,9,8,7,6,5,4,3,2",
        help='Comma-separated class names in EXACT training order'
    )
    p.add_argument('--smooth', type=int, default=15, help='Temporal smoothing window (frames)')
    p.add_argument('--conf_thresh', type=float, default=0.70, help='Min confidence to display')
    p.add_argument('--mirror', action='store_true', help='Flip camera horizontally (selfie view)')
    p.add_argument('--pad_square', action='store_true', help='Pad to square instead of center-crop')
    p.add_argument('--imagenet_norm', action='store_true', help='Use ImageNet mean/std normalization (enable only if training used it)')
    return p.parse_args()



def softmax(x):
    x = x - np.max(x)
    e = np.exp(x, dtype=np.float32)
    return e / np.sum(e)


def preprocess(frame, size, pad_square=False, imagenet_norm=False, expect_nchw=True):
    # Make square: center-crop (default) or pad
    h, w = frame.shape[:2]
    if pad_square:
        side = max(h, w)
        sq = np.zeros((side, side, 3), dtype=frame.dtype)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        sq[y0:y0+h, x0:x0+w] = frame
    else:
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        sq = frame[y0:y0+side, x0:x0+side]

    img = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    if expect_nchw:
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = img[None, ...]                # NCHW
    else:
        img = img[None, ...]                # NHWC

    return img


def infer_shape(session):
    inp = session.get_inputs()[0]
    name = inp.name
    shape = inp.shape  # e.g., [1, 3, 224, 224] or [1, 224, 224, 3]
    expect_nchw = True
    if len(shape) == 4:
        # Decide based on channel position
        # If second dim is 3, assume NCHW; if last dim is 3, assume NHWC
        if shape[1] == 3:
            expect_nchw = True
        elif shape[-1] == 3:
            expect_nchw = False
    return name, expect_nchw


def main():
    args = parse_args()
    
    print('Labels order:', args.labels)
    for i, n in enumerate(args.labels.split(',')):
        print(i, '->', n)

    labels = args.labels.split(',')
    num_classes = len(labels)

    # ONNX Runtime session
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name, expect_nchw = infer_shape(sess)
    out_name = sess.get_outputs()[0].name
    print(f'Input name={in_name}, NCHW={expect_nchw}')

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print('Error: cannot open camera')
        return

    # Store full probability vectors for smoothing
    smooth_q = deque(maxlen=max(args.smooth, 1))
    t_last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # Preprocess to model input
        inp = preprocess(
            frame, args.imgsz,
            pad_square=args.pad_square,
            imagenet_norm=args.imagenet_norm,
            expect_nchw=expect_nchw
        )

        # Inference
        logits = sess.run([out_name], {in_name: inp})[0].squeeze()
        # Support both raw logits and already-softmaxed outputs
        probs = logits if np.all((logits >= 0) & (logits <= 1)) and abs(np.sum(logits) - 1) < 1e-3 else softmax(logits)

        # Temporal probability averaging
        smooth_q.append(probs)
        avg_probs = np.mean(np.stack(smooth_q, axis=0), axis=0)
        top_idx = int(np.argmax(avg_probs))
        top_p = float(avg_probs[top_idx])

        # FPS (EMA)
        now = time.time()
        dt = now - t_last
        t_last = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Overlay
        label = labels[top_idx] if 0 <= top_idx < num_classes else '?'
        text = f'{label}  {top_p:.2f}  {fps:.1f} FPS'
        color = (40, 200, 40) if top_p >= args.conf_thresh else (60, 60, 200)
        print(f'PRED {label} | conf={top_p:.2f} | fps={fps:.1f}', flush=True)
        write_stats(label, top_p, fps)


        cv2.rectangle(frame, (8, 8), (360, 50), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        cv2.imshow('ONNX Runtime Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

