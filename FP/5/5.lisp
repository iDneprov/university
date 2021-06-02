(defclass cart ()
 ((x :initarg :x :reader x)
  (y :initarg :y :reader y)))

(defmethod print-object ((c cart) stream)
  (format stream "[CART x ~d y ~d]"(x c) (y c) ))

(defclass polar ()
 ((radius :initarg :radius :accessor radius)
  (angle  :initarg :angle  :accessor angle)))

(defmethod print-object ((p polar) stream)
  (format stream "[POLAR radius ~d angle ~d]"
    (radius p) (angle p)))

(defclass triangle ()
 ((dot1 :initarg :dot1 :reader dot1)
   (dot2 :initarg :dot2 :reader dot2)
   (dot3 :initarg :dot3 :reader dot3)))



(defmethod print-object ((triang triangle) stream)
  (format stream "[TRIANG ~s ~s ~s]"
    (dot1 triang) (dot2 triang) (dot3 triang)))

(defmethod x ((p polar))
  (* (radius p) (cos (angle p))))

(defmethod y ((p polar))
  (* (radius p) (sin (angle p))))

(defgeneric polarToCart (arg)
 (:method ((c cart)) c)
 (:method ((p polar))
  (make-instance 'cart
    :x (x p)
    :y (y p))))

(defun sqare (x)
  (* x x))

(defmethod calcDist ((c1 cart) (c2 cart))
  (sqrt (+ (sqare (abs (- (x c1) (x c2))) ) (sqare (abs (- (y c1) (y c2)))))))

(defmethod calcLen ((p1 polar) (p2 polar))
  (let ((dot1 (polarToCart p1)))
  (let ((dot2 (polarToCart p2)))
  (calcDist dot1 dot2))))  

(defmethod calcLen ((c1 cart) (c2 cart))
  (calcDist c1 c2)) 

(defmethod calcLen ((c1 cart) (p2 polar))
  (let ((dot2 (polarToCart p2)))
  (calcDist c1 dot2)))

(defmethod calcLen ((p1 polar) (c2 cart))
  (let ((dot1 (polarToCart p1)))
  (calcDist dot1 c2)))

(defun прямоугольный-p (triang)
  (let ((a (calcLen (dot1 triang) (dot2 triang))))
  (let ((b (calcLen (dot2 triang) (dot3 triang))))
  (let ((c (calcLen (dot3 triang) (dot1 triang))))
  (cond
    ((< (abs (- (sqare a) (+ (sqare b) (sqare c)))) 0.0001)  T)
    ((< (abs (- (sqare b) (+ (sqare c) (sqare a)))) 0.0001)  T)
    ((< (abs (- (sqare c) (+ (sqare a) (sqare b)))) 0.0001)  T))))))



(setq triang0 (make-instance 'triangle
  :dot1 (make-instance 'cart :x 1 :y 0)
  :dot2 (make-instance 'cart :x 0 :y 1)
  :dot3 (make-instance 'cart :x 0 :y 0)))

(print (прямоугольный-p triang0))


(setq triang1 (make-instance 'triangle
  :dot1 (make-instance 'polar :radius 5 :angle 0)
  :dot2 (make-instance 'cart :x 0 :y 5)
  :dot3 (make-instance 'polar :radius 0 :angle 0)))

(print (прямоугольный-p triang1))

(setq triang2 (make-instance 'triangle
  :dot1 (make-instance 'polar :radius 5 :angle 0)
  :dot2 (make-instance 'polar :radius 4 :angle 1)
  :dot3 (make-instance 'polar :radius 0 :angle 0)))

(print (прямоугольный-p triang2))

(setq triang3 (make-instance 'triangle
  :dot1 (make-instance 'cart :x 100 :y 0)
  :dot2 (make-instance 'cart :x 7 :y 6)
  :dot3 (make-instance 'polar :radius 1 :angle (/ pi 4))))

(print (прямоугольный-p triang3))

