(defun pascal-triangle (n)
   (print-rows n '(1)))

(defun print-rows (n l)
   (when (< 0 n)
       (print l)
       (print-rows (1- n) (cons 1 (newrow l)))))

(defun newrow (l)
   (if (> 2 (length l))
      '(1)
      (cons (+ (car l) (cadr l)) (newrow (cdr l)))))