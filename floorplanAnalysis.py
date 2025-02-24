import cv2
import numpy as np
import pytesseract
from scipy.stats import mode
import os
import yaml
from yaml.loader import SafeLoader 

pytesseract.pytesseract.tesseract_cmd = "C:/Users/Jackw/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
class analysis:
    def __init__(self): 
        pass
    
    
    def get_analysis(search_for):
        def align_to_wall(wall_start, wall_end, door_center_x, door_center_y):
            """Align a door to be flush with a wall segment."""
            x1, y1 = wall_start
            x2, y2 = wall_end

            # If the wall is vertical, align the door's x-coordinate
            if x1 == x2:
                aligned_x = x1
                aligned_y = door_center_y

            # If the wall is horizontal, align the door's y-coordinate
            elif y1 == y2:
                aligned_x = door_center_x
                aligned_y = y1

            # For diagonal walls, align to the closest point on the line segment
            else:
                # Calculate the closest point on the line to the door center
                dx = x2 - x1
                dy = y2 - y1
                t = ((door_center_x - x1) * dx + (door_center_y - y1) * dy) / (dx ** 2 + dy ** 2)
                t = max(0, min(1, t))  # Clamp t to the segment [0, 1]
                aligned_x = x1 + t * dx
                aligned_y = y1 + t * dy

            return int(aligned_x), int(aligned_y)
        def sharpen_image(image):
            # Convert the image to grayscale if it's not already
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and emphasize edges
            blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Apply Laplacian kernel for sharpening
            sharpened = cv2.addWeighted(image, 2, blurred, -1, 0)

            return sharpened


        def get_room_type(text):
            # Add conditions based on the extracted text to determine room type
            # text lower so that it isn't case sensitive
            text_lower = text.lower()
            if "bedroom" in text_lower:
               # print("identified Text:", text)  # Print the unidentified text for debugging
                return "bedroom"
            elif any(keyword in text_lower for keyword in ["living room", "living area", "lounge", "office", "sitting area"]):
              #  print("identified Text:", text)  # Print the unidentified text for debugging
                return "living room"
            elif any(keyword in text_lower for keyword in ["bathroom", "wc", "toilet", "en-suite", "en suite", "shower"]):
               # print("identified Text:", text)  # Print the unidentified text for debugging
                return "bathroom"
            elif any(keyword in text_lower for keyword in ["dining", "kitchen"]):
               # print("identified Text:", text)  # Print the unidentified text for debugging
                return "food"
            elif any(keyword in text_lower for keyword in ["store", "utility", "toilet"]):
               # print("identified Text:", text)  # Print the unidentified text for debugging
                return "utility"
            else:
                return "default"


        def find_rooms(img, noise_removal_threshold=25, corners_threshold=0.1,
                    room_closing_max_length=200, gap_in_wall_threshold=600):
            """

            :param img: grey scale image of rooms, already eroded and doors removed etc.
            :param noise_removal_threshold: Minimal area of blobs to be kept.
            :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
            :param room_closing_max_length: Maximum line length to add to close off open doors.
            :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
            :return: rooms: list of numpy arrays containing boolean masks for each detected room
            colored_house: A colored version of the input image, where each room has a random color.
            """
            assert 0 <= corners_threshold <= 1
            # Remove noise left from door removal

            img[img < 128] = 0
            img[img > 128] = 255
            contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(img)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > noise_removal_threshold:
                    cv2.fillPoly(mask, [contour], 255)

            img = ~mask

            # Detect corners (you can play with the parameters here)
            dst = cv2.cornerHarris(img ,2,3,0.04)
            dst = cv2.dilate(dst,None)
            corners = dst > corners_threshold * dst.max()

            # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
            # This gets some false positives.
            # You could try to disallow drawing through other existing lines for example.
            for y, row in enumerate(corners):
                x_same_y = np.argwhere(row)
                for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
                    if x2[0] - x1[0] < room_closing_max_length:
                        color = 0
                        cv2.line(img, (x1[0], int(y)), (x2[0], int(y)), color, 1)


            for x, col in enumerate(corners.T):
                y_same_x = np.argwhere(col)
                for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
                    if y2[0] - y1[0] < room_closing_max_length:
                        color = 0
                        cv2.line(img, (int(x), y1[0]), (int(x), y2[0]), color, 1)



            # Mark the outside of the house as black
            contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [biggest_contour], 255)
            img[mask == 0] = 1
            epsilon = 0.01 * cv2.arcLength(biggest_contour, True)  # You can adjust the epsilon value
            approx_vertices = cv2.approxPolyDP(biggest_contour, epsilon, True)

            vertices = [(int(vertex[0][0]), int(vertex[0][1])) for vertex in approx_vertices]

            # Draw the approximated contour on the image (for visualization)
            cv2.drawContours(img, [approx_vertices], -1, (0, 255, 0), 3)

            # Print the vertices
          #  print("Vertices of the biggest contour:")
            for vertex in approx_vertices:...
         #       print(vertex[0])
            # Find the connected components in the house
            ret, labels = cv2.connectedComponents(img)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            unique = np.unique(labels)
            rooms = []
            for label in unique:
                component = labels == label
                if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
                    color = 0
                else:
                    rooms.append(component)
                    color = np.random.randint(0, 255, size=3)
                img[component] = color

            return rooms, img, vertices
        # Read the original color image
        img = cv2.imread("floorplan.jpeg", cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to obtain a binary image
        _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)


        # Define the structuring element. Here we use a 3x3 rectangle
        kernel = np.ones((3, 3), np.uint8)

        # Perform erosion
        erosion = cv2.dilate(binary_img, kernel, iterations=2)
        erosion = cv2.erode(erosion, kernel, iterations=3)
        erosion = cv2.dilate(erosion, kernel, iterations=1)

        sharpened_image = sharpen_image(img)

        # Find rooms and obtain the colored house image
        rooms, colored_house, walls = find_rooms(erosion.copy())

       # print(walls)
        # Perform OCR on the original image to extract text
        text = pytesseract.image_to_string(img, config='--psm 11')

        # Print the OCR output for debugging
        #print("OCR Output:", text)

        # Overlay the colored rooms on the original image

        alpha = 0.6  # Adjust the alpha value for a darker overlay
        result = cv2.addWeighted(img, alpha, colored_house, 1 - alpha, 0)
        
        
        room_data = []
        for i, room in enumerate(rooms):
            # Extract the specific region for the room
            room_region = 255 * room.astype(np.uint8)

            # Create a mask for the room region
            room_mask = np.zeros_like(img_gray)
            room_mask[room] = 255

            # Extract the room's ROI from the original image
            room_roi = cv2.bitwise_and(img_gray, img_gray, mask=room_mask)

            # Perform OCR on the room's ROI 
            # PSM 11 used as it yielded the most accurate results compared to psm 6 being the next most accurate
            room_text = pytesseract.image_to_string(room_roi, config='--psm 11')
            print(room_text)

            # Determine room type based on the extracted text for the specific room
            room_type = get_room_type(room_text)
            #print(f"Room {i + 1} Type: {room_type}")

            # Get the dimensions of the room
            contours, _ = cv2.findContours(room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                #print(f"Room {i + 1} Dimensions: Width={w} pixels, Height={h} pixels")
                
                # Get the coordinates of each vertex in the room
                vertices = largest_contour.reshape(-1, 2)
                for vertex in vertices:...
                    #print(f"Vertex: ({vertex[0]}, {vertex[1]})")
                room_data.append([room_type, vertices])
                    


           # print("------")

        # load yaml file and yolo model
        with open('yolo\pascal data\data.yaml') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        labels = data_yaml['names']

        yolo = cv2.dnn.readNetFromONNX('yolov5/runs/train/exp/weights/last.onnx')
        # set package to CPU 
        yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # load image
        # from image get the detections
        img = cv2.imread('floorplan.jpeg')
        image = result.copy()

        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        row, col, d = image.shape

        # convert image into square matirix, this is because the yolo model wants square images
        max_rc = max(row, col) # get the max value for rows and colums
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8) # create a square image that is blank

        input_image[0:row, 0:col] = image # overlay the two images to get a square image with the original image inside

        #get predictions from square array 
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (640,640), swapRB=True, crop=False)
        yolo.setInput(blob)
        preds = yolo.forward() # get predictions

        # non maximum suppresion to make sure bounding boxes are correct, removes duplicates and give only good probabality score
        # filter on probablility score and confidence scorre i.e if confidence is less than 0.4 then filter it out
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # calculate width and height of image
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/640
        y_factor = image_h/640 # these are expected to be the same as the image should be square

        door_sizes =[]
        ds=1

        door_widths = []

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of detection of object
            if confidence > 0.5:
                class_score = row[5:].max() # take maximum probability of object have 8 objects of all eight objects take the one it is most likely to be
                class_id = row[5:].argmax() # get index position at which maximum probability occurs
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    #construct bounding box from four values
                    # Left, Top, Width, Height
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    
                    box = np.array([left,top,width,height])
                    if labels[class_id] == 'door':
                        door_sizes.append(width*height)
                        if height < width:
                            door_widths.append(height)
                        else:
                            door_widths.append(width)
                        ds = ds + 1  

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # cleaning data
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # apply NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.3).flatten() # this gives index positions


        #print(door_sizes)


        average_area = sum(door_sizes) / len(door_sizes)
        #print(average_area)
        # draw the bounding box
        for ind in index:
            # extract bounding boxes
            x,y,w,h = boxes_np[ind]
            # take confidences
            bb_conf = confidences_np[ind]
            # take classes
            classes_id  = classes[ind]
            class_name = labels[classes_id]
            width = int(w*x_factor)
            height = int(h*y_factor)
            if class_name == 'door': # rule to make sure doors are consistent
                if (width * height) > average_area * 0.8 or (width * height) < average_area * 1.15:
                    text = f'{class_name}: {bb_conf}%'
            
                    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.rectangle(image, (x,y-30),(x+w,y),(255,255,255), -1)
                    cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1)
            else:
                text = f'{class_name}: {bb_conf}%'
                
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.rectangle(image, (x,y-30),(x+w,y),(255,255,255), -1)
                cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1)
                

        cv2.imshow('original', img)
        cv2.imshow('Binary', binary_img)
        cv2.imshow('noise reduction', erosion)
        cv2.imshow('sharpened', sharpened_image)
        cv2.imshow('result', result)
        cv2.imshow('yolo_prediction', image)

        #cv2.waitKey(0)
        #cv2.destroyAllWindowsa

        if search_for == 'walls':
            #print(walls)
            return walls
        elif search_for == 'scale':
            # 9000 because it would be 900mm wide (90cm) therefore the number of pixels in a that area is equal to 9000mm which gives a scale of px to mm
            # the reason for average area is because doors are translated 90 degrees meaning that they will have different width and height on the orientation 
            # but should still cover the same area although this does take some assumptions it is a reliable way to get scale despite not being totally accurate
            # ignore above changing scale from mm to cm because want to save processing power :)
            
            scale_factor  =  (sum(door_widths)/ len(door_widths)) / 90
            print("there are ", scale_factor, "pixels per mm")
            return scale_factor
        
        if search_for == 'doors':
            # Store flushed door data
            flushed_doors = []

            for ind in index:
                # Extract bounding box
                x, y, w, h = boxes_np[ind]
                classes_id = classes[ind]
                class_name = labels[classes_id]

                if class_name == 'door':  # Process only doors
                    door_center_x = x + w // 2
                    door_center_y = y + h // 2

                    # Find the closest wall
                    closest_wall = None
                    min_distance = float('inf')
                    for i in range(len(walls) - 1):
                        wall_start = walls[i]
                        wall_end = walls[i + 1]

                        # Compute the distance from the door's center to the wall segment
                        px, py = door_center_x, door_center_y
                        x1, y1 = wall_start
                        x2, y2 = wall_end
                        distance = np.abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / (
                            ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
                        )

                        # Update if this wall is closer
                        if distance < min_distance:
                            min_distance = distance
                            closest_wall = (wall_start, wall_end)

                    # Align the door coordinates to the closest wall
                    if closest_wall:
                        wall_start, wall_end = closest_wall
                        aligned_x, aligned_y = align_to_wall(wall_start, wall_end, door_center_x, door_center_y)

                        # Add the flushed door information
                        flushed_doors.append({
                            'door_coordinates': (aligned_x, aligned_y, w, h),
                            'closest_wall': closest_wall,
                            'distance_to_wall': min_distance
                        })

            return flushed_doors
        
        if search_for == "rooms":
            return room_data



        
        