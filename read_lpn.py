import math


# license plate type classification helper function
def linear_equation(x1, y1, x2, y2) :
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b


def check_point_linear(x, y, x1, y1, x2, y2) :
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return (math.isclose(y_pred, y, abs_tol=3))


def read_plate(yolo_license_plate, im) :
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()
    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10 :
        return "unknown"
    center_list = []
    y_mean = 0
    y_sum = 0
    list_conf = []
    for bb in bb_list :
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        list_conf.append(bb[4])
        y_sum += y_c
        center_list.append([x_c, y_c, bb[-1]])
    sum_conf = sum(list_conf)
    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list :
        if cp[0] < l_point[0] :
            l_point = cp
        if cp[0] > r_point[0] :
            r_point = cp
    for ct in center_list :
        if l_point[0] != r_point[0] :
            if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False) :
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))
    size = results.pandas().s

    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2" :
        for c in center_list :
            if int(c[1]) > y_mean :
                line_2.append(c)
            else :
                line_1.append(c)
        for l1 in sorted(line_1, key=lambda x : x[0]) :
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key=lambda x : x[0]) :
            license_plate += str(l2[2])
    else :
        for l in sorted(center_list, key=lambda x : x[0]) :
            license_plate += str(l[2])
    return license_plate, sum_conf


def get_car(license_plate, vehicle_track_ids) :
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)) :
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2 :
            car_indx = j
            foundIt = True
            break

    if foundIt :
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
