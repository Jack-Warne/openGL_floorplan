    text = f'{class_name}: {bb_conf}%'
        
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.rectangle(image, (x,y-30),(x+w,y),(255,255,255), -1)
        cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1)
    