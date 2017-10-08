cmake_minimum_required(VERSION 3.3)
project(FaceLandmark)
find_package( OpenCV )
find_package( OpenBLAS)
find_package(PythonInterp)
find_package(PythonLibs)
find_package(Boost COMPONENTS system python filesystem REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES} ${PYTHON_LINBRARIES})

set(SOURCE_FILES featpyramid.cpp model.cpp NPDDetector.cpp NPDModel.cpp Pose.cpp shapeRegression.cpp stdafx.cpp config.h mtimes.h targetver.h _type.h imtransform.cpp)
add_executable(face_DLRC ${SOURCE_FILES} face_DLRC.cpp )
target_link_libraries(face_DLRC ${OpenCV_LIBS} ${OpenBLAS_LIBRARIES})
add_executable(batchrc batch_rotate_crop.cpp ${SOURCE_FILES} batch_DLRC.cpp)
target_link_libraries(batchrc ${OpenCV_LIBS} ${OpenBLAS_LIBRARIES}  )
add_executable(list_crop List_crop.cpp ${SOURCE_FILES})
target_link_libraries(list_crop ${OpenCV_LIBS} ${OpenBLAS_LIBRARIES}  )
add_executable(batch_DLRC batch_DLRC.cpp ${SOURCE_FILES})
target_link_libraries(batch_DLRC ${OpenCV_LIBS} ${OpenBLAS_LIBRARIES}  )