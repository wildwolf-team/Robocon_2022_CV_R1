add_library(Camera mv_video_capture.cpp)

#fmt
find_package(fmt REQUIRED)
target_link_libraries(Camera fmt::fmt)

#MVSDK
add_library(MVSDK SHARED IMPORTED)
set(MIMDVISION_SDK_LIBRARY /lib/libMVSDK.so)
set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
target_link_libraries(Camera MVSDK)