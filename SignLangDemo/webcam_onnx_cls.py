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
    main()
