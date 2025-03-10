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


# detect character and number in license plate
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

    # 1 line plates and 2 line plates
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

    # if frame_nmr % 10 == 0 :  # Chỉ thực hiện mỗi 10 frame
    #     if frame_nmr in results_best :  # Kiểm tra frame_nmr có tồn tại không
    #         for car_id in results[frame_nmr] :
    #             if car_id in results_best[frame_nmr] :  # Kiểm tra car_id có tồn tại không
    #                 if 'vehicle' in results_best[frame_nmr][car_id] and 'license_plate' in results_best[frame_nmr][
    #                     car_id] :
    #                     # Lấy thông tin xe
    #                     vehicle_bbox = results_best[frame_nmr][car_id]['vehicle']['bbox']
    #
    #                     # Lấy thông tin biển số
    #                     license_plate_data = results_best[frame_nmr][car_id]['license_plate']
    #
    #                     # Kiểm tra 'text_score' tồn tại và không rỗng
    #                     if 'text_score' in license_plate_data and license_plate_data['text_score'] :
    #                         best_score_index = license_plate_data['text_score'].index(
    #                             max(license_plate_data['text_score']))
    #
    #                         # Lấy thông tin biển số tốt nhất dựa vào điểm số cao nhất
    #                         best_license_plate = {
    #                             'bbox' : license_plate_data['bbox'],
    #                             'text' : license_plate_data['text'][
    #                                 best_score_index] if 'text' in license_plate_data else '',
    #                             'text_score' : max(license_plate_data['text_score'])
    #                         }
    #
    #                         # Cập nhật kết quả tốt nhất
    #                         results_best[frame_nmr][car_id] = {
    #                             'vehicle' : {'bbox' : vehicle_bbox},
    #                             'license_plate' : best_license_plate
    #                         }
    #                     else :
    #                         print(f"Warning: 'text_score' missing or empty for car_id {car_id} in frame {frame_nmr}")
    #                 else :
    #                     print(f"Warning: 'vehicle' or 'license_plate' missing for car_id {car_id} in frame {frame_nmr}")
    #             else :
    #                 print(f"Warning: car_id {car_id} not found in results_best[{frame_nmr}]")
    #     else :
    #         print(f"Warning: frame_nmr {frame_nmr} not found in results_best")
