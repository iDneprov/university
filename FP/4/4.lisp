(defun lowerChar (char)
  (position char "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxy"))

(defun changeStarWord (text pos)
  (loop for a from 0 to pos
    do (if (lowerChar (char text a)) (setf (char text a) #\3) ())
  )
text)

(defun changeStandartWord (text)
  (loop for a from 0 to (- (length text) 1)
    do (if (lowerChar (char text a)) (setf (char text a) #\3) ())
  )
text)

(defun changeWord (text &aux (pos (position #\* text)))
  (if pos (changeStarWord text pos) (changeStandartWord text)
)text)

(defun haveStar (text)
  (loop for a from 0 to (- (length text) 1)
    do (if (position #\* (elt text a)) (return a))
  )
)

(defun lab4 (text)
  (if (haveStar text) 
  (loop for a from 0 to (haveStar text)
    do (changeWord (elt text a))
  )
  ())
text)


(lab4 '("we" "rfG .gd" "ww" "w*w" "we" "rfG .gd"))
(lab4 '("we" "rfG .gd" "ww" "ww" "we" "rfG .gd"))
(lab4 '("Првыфпв34RU" "а   .;67 ап ППП " "ww" "аА*бБ" "Првыфпв34RU" "а   .;67 ап ППП "))
(lab4 '("ПроВерка"))
(lab4 '("Про*Верка"))
