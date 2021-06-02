(defun reverse-list (w &optional acc)
  (if w (reverse-list (cdr w) (cons (car w) acc)) acc))

(defun is-decreasing (w)
  (cond ((null (cdr w)) t)
    ((<= (car w) (cadr w)) nil)
    ((is-decreasing (cdr w)))))

(defun decreasing (l)
  (if (is-decreasing l)
    l
    (reverse-list l)))