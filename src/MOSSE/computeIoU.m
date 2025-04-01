function iou = computeIoU(boxA, boxB)
    % computeIoU calculates the Intersection over Union of two boxes.
    % boxA and boxB are in [x, y, width, height] format.
    xA = max(boxA(1), boxB(1));
    yA = max(boxA(2), boxB(2));
    xB = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    yB = min(boxA(2) + boxA(4), boxB(2) + boxB(4));
    interArea = max(0, xB - xA) * max(0, yB - yA);
    boxAArea = boxA(3) * boxA(4);
    boxBArea = boxB(3) * boxB(4);
    iou = interArea / (boxAArea + boxBArea - interArea);
end