import os, sys, pdb, sqlite3 as sq
import torch

aflw_sq  = sq.connect('aflw.sqlite')
aflw_cur = aflw_sq.cursor()
print ("Succesfully open the aflw.sqlite")


# show the tables in the database
aflw_cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_name = aflw_cur.fetchall()
for item in table_name:
  print (item)

# fetch image_name, face_rect and feature coordinates from db
faces = aflw_cur.execute("SELECT * FROM Faces")
face_ids = faces.fetchall();

faces = {}
for i in range(len(face_ids)): 
    print ('{:04d}/{:04d} : {:} : {:}'.format(i, len(face_ids), face_ids[i][0], face_ids[i][1]))
    # get face_id and file_id
    face_id = face_ids[i][0]
    file_id_sqlite = "SELECT file_id FROM Faces WHERE face_id ='" + str(face_id) + "'"
    file_id = aflw_cur.execute(file_id_sqlite).fetchall()
    file_id = file_id[0][0]
    if len(file_id) < 1:
        continue
    
    # get file_path
    face_name_query = "SELECT filepath FROM FaceImages WHERE file_id = '"+ file_id + "'"
    face_name = aflw_cur.execute(face_name_query).fetchall()
    face_name = face_name[0][0]

    # rect
    feature_rect_query = "SELECT FaceRect.x,FaceRect.y,FaceRect.w,FaceRect.h FROM FaceRect WHERE face_id ='" + str(face_id) + "'"
    feature_rect = aflw_cur.execute(feature_rect_query).fetchall()

    if len(feature_rect) < 1: continue
    feature_rect = feature_rect[0]
    x = feature_rect[0]
    y = feature_rect[1]
    w = feature_rect[2]
    h = feature_rect[3]

    feature_pose_query = "SELECT FacePose.yaw,FacePose.pitch,FacePose.roll FROM FacePose WHERE face_id ='" + str(face_id) + "'"
    feature_pose = aflw_cur.execute(feature_pose_query).fetchall()

    if face_name in faces:
      faces[face_name].append( {'rect': feature_rect, 'pose':feature_pose} )
    else:
      faces[face_name] = [ {'rect': feature_rect, 'pose':feature_pose} ]

torch.save(faces, 'aflw-sqlite.pth')
