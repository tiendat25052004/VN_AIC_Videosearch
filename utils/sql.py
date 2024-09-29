import mysql.connector
from mysql.connector import Error

# Thay thế các thông số dưới đây bằng thông tin thực tế của bạn
host = '104.214.176.95'
port = 3306
username = 'root'     # Tên đăng nhập MySQL của bạn
password = 'sang'      # Mật khẩu MySQL của bạn
database = 'object' # Tên cơ sở dữ liệu bạn muốn kết nối


def query_all_object(cursor, class_names: list, bottoms, tops, lefts, rights):
    # Chuyển đổi class_names thành class_ids
    cursor.execute("SELECT id FROM class WHERE name IN (%s)" % ','.join(['%s'] * len(class_names)), class_names)
    class_ids = [row[0] for row in cursor.fetchall()]
    print(class_ids)

    # Tạo điều kiện cho câu lệnh SQL
    conditions = []
    for i in range(len(class_ids)):
        conditions.append(f"(tb.row BETWEEN %s AND %s AND tb.col BETWEEN %s AND %s)")
    
    # Kết hợp các điều kiện
    where_clause = " OR ".join(conditions)
    
    select_query = f"""
    SELECT tb.frame_id
    FROM bbox tb
    JOIN class c ON tb.class_id = c.id
    WHERE c.id IN ({','.join(['%s'] * len(class_ids))}) AND ({where_clause})
    GROUP BY tb.frame_id
    HAVING COUNT(DISTINCT c.name) >= %s
    """
    
    # Chuẩn bị tham số cho truy vấn
    params = []
    for i in range(len(class_ids)):
        params.extend([bottoms[i], tops[i], lefts[i], rights[i]])
    params.extend(class_ids)  # Các class_id
    params = class_ids + params
    
    print("Generated SQL Query:")
    print(select_query)
    
    # In các tham số của truy vấn
    print("Query Parameters:")
    print(params)

    cursor.execute(select_query, params)

    # Lấy kết quả
    result = cursor.fetchall()  # Lấy tất cả các kết quả
    frame_ids = [row[0] for row in result] if result else []
    
    return set(frame_ids)  # Trả về tập hợp các frame_id

def handle_object_filter(object_input):
    try:
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # print("Kết nối thành công!")

            # Gọi biến đầu vào
            class_names = []  
            tops = []              
            lefts = []             
            rights = []            
            bottoms = []
            for object_item in object_input:
                class_names.append(object_item["name"])
                tops.append(object_item["topleft"][0])
                lefts.append(object_item["topleft"][1])
                bottoms.append(object_item["botright"][0])
                rights.append(object_item["botright"][1])
            

            #query
            common_frame_ids = query_all_object(cursor, class_names, bottoms, tops, lefts, rights)
            return common_frame_ids
            print("Các frame_ids thỏa mãn tất cả các class_names:", sorted(common_frame_ids))


    except Error as e:
        print("Lỗi khi kết nối:", e)

    finally:
        # Đảm bảo đóng con trỏ và kết nối
        if 'cursor' in locals() and cursor:  # Kiểm tra nếu cursor được định nghĩa
            cursor.close()
        if connection.is_connected():
            connection.close()
            print("Kết nối đã được đóng.")