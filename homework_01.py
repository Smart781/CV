import cv2
import numpy as np

points = []

color_jitter_params = {
    'alpha': 0,  
    'beta': 100,   
    'gamma': 100,  
    'delta': 100   
}

flag = False

def create_control_panel():
    cv2.namedWindow('Color Jitter')
    cv2.createTrackbar('alpha', 'Color Jitter', 0, 360, update_color_jitter)
    cv2.createTrackbar('beta%', 'Color Jitter', 100, 100, update_color_jitter)
    cv2.createTrackbar('gamma%', 'Color Jitter', 100, 100, update_color_jitter)
    cv2.createTrackbar('delta%', 'Color Jitter', 100, 100, update_color_jitter)

def get_jitter_params():
    alpha = cv2.getTrackbarPos('alpha', 'Color Jitter')
    beta = cv2.getTrackbarPos('beta%', 'Color Jitter') / 100.0
    gamma = cv2.getTrackbarPos('gamma%', 'Color Jitter') / 100.0
    delta = cv2.getTrackbarPos('delta%', 'Color Jitter') / 100.0
    return alpha, beta, gamma, delta

def update_color_jitter(val):
    pass

def mouse_callback(event, x, y, flags, param):
    global points
    global flag

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 4:
            flag = True
            points = []
        points.append([x, y])

def draw_points(image):
    global points
    global flag

    for x, y in points:
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)
        
    if len(points) == 4:
        # if flag:
        #     create_control_panel()
        #     flag = False

        image = square(image)
        a, b, g, d = get_jitter_params()
        augmented_img = color_jitter(image, alpha=a, beta=b, gamma=g, delta=d)
        cv2.imshow('ColorJitter', augmented_img)


def square(image):
    src_pts = np.float32(points)
    h = 360
    w = 680

    dst_img = np.zeros((h, w, 3), dtype=np.uint8)

    dst_pts = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts) 

    dst_img = cv2.warpPerspective(image, H, (w, h), cv2.INTER_LINEAR)

    return dst_img

def color_jitter(image, alpha=0, beta=1.0, gamma=1.0, delta=1.0):
    img = image.astype(np.float32) / 255.0
    
    if alpha != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + alpha) % 360 
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    if beta != 1.0:
        Y = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        Y = np.stack([Y, Y, Y], axis=-1)
        img = beta * img + (1 - beta) * Y
    
    if gamma != 1.0:
        Y = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        mean = np.mean(Y, axis=(0, 1), keepdims=True)
        img = gamma * img + (1 - gamma) * mean
    
    if delta != 1.0:
        img = img * delta
    
    return np.clip(img * 255, 0, 255).astype(np.uint8)
    



def main():
    global points

    window_name = 'Image'  
    video = cv2.VideoCapture(0)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback) 

    create_control_panel()

    while True:
        ret, image = video.read()
        if not ret:
            break
        
        draw_points(image)
        cv2.imshow(window_name, image)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()