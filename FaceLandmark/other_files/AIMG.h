#ifndef AMG_H
#define AMG_H

#include <windows.h>

/**
 * The version of this SDK.
 */
#define AIM_VERSION "1.0.2"

/** 
 * The length of face feature.
 */
#define FEATURE_SIZE 4004

/** 
 * The length of face signature.
 */
#define SIGNATURE_SIZE 332

/**
 * Face engine object.
 */
DECLARE_HANDLE(HAIM);

/**
 * The max count of landmarks.
 */
#define MAX_LANDMARKS_NUM 27

/**
 * A facial landmark.
 */
typedef struct {
	int id;
	float x;
	float y;
} TLandmark;

/**
 * The landmarking result.
 */
typedef struct {
	int count;
	int view;		/*< 0: (-90, -30), 1: (-30, 0), 2: (0, 30), 3: (30, 90) */
	TLandmark pts[MAX_LANDMARKS_NUM];
} TLandmarks;

/**
 * The result of processed faces.
 */
typedef struct {
	RECT rtFace;								/*< coordinate of face in the image */
	POINT ptLeftEye;							/*< coordinate of the left eye */ 
	POINT ptRightEye;							/*< coordinate of the right eye */
	float confidence;							/*< confidence of face detector */
	float frontalDirection[3];					/*< direction of face normal */
	float quality;								/*< quality of face image, e.g., blur, noise, and so on */
	TLandmarks landmarks;						/*< facial landmarks */
	unsigned char feature[FEATURE_SIZE];		/*< feature template */
} TFaceResult;

/**
 * The parameters for face detection and quality assessment.
 * Only the faces satisfy the following constrains will be output
 * 1. the eye distance > minEyeDistance;
 * 2. the eye distance < maxEyeDistance;
 * 3. -roll < the roll angle of face < roll;
 * 4. the confidence of face < confidence;
 * 5. the frontal angle of face < frontalAngle;
 * 6. the quality of face < quality (0 denotes the best quality, 100 denotes the worst quality).
 */
typedef struct {
	float minEyeDistance;		/*< positive, default: 15 */
	float maxEyeDistance;		/*< positive, > minEyeDistance, default: 10000 */
	float roll;					/*< from 0 to 180 degree, default: 0 */
	float confidence;			/*< from 0 to 1, default: 0.2 */
	float frontalAngle;			/*< from 0 to 90 degree,  default: 90 */
	float quality;				/*< from 0 to 100,  default: 100 */
	int sortType;				/*< 0: sort by confidence, 1: by size, default: 0 */
} TFaceParams;

/**
 * Create face engine.
 * @return if success return the handle of face engine, else return NULL.
 */
typedef HAIM (*TAimCreate)();

/** 
 * Free face engine.
 * @param handle the handle of face engine.
 */
typedef void (*TAimFree)(HAIM handle);

/**
 * Set parameters for face detection and quality assessment.
 * @param handle the handle of face engine.
 * @param params the parameters for face detection.
 * @return if success return 0, else return the error code.
 */
typedef int (*TAimSetParams)(HAIM handle, const TFaceParams* params);

/**
 * Detect face and extract feature with parameters.
 * @param handle the handle of face engine.
 * @param image image data in BGR24 or Gray format.
 * @param width the width of image.
 * @param height the height of image.
 * @param pitch the byte offset between two lines of image.
 * @param bpp 8: gray image, 24: BGR image. 
 * @param maxCount the maximum number of faces we want to detect in the image.
 * @param results the properties of detected faces and their features.
 * @return the number of detected faces. 
 */
typedef int (*TAimProcess)(HAIM handle, const unsigned char* image, int width, int height, int pitch, int bpp,
		 int maxCount, TFaceResult* results);

/**
 * Match two features. 
 * Note: this function is totally thread-safe.
 * @param handle the handle of face engine.
 * @param fea1 feature data of a face image.
 * @param fea2 feature data of another face image.
 * @return return the similarity score of two features (0 to 1).
 */
typedef float (*TAimMatch)(HAIM handle, const unsigned char* fea1, const unsigned char* fea2);

/**
 * Extract feature based on manually labeled landmarks.
 * @param handle the handle of face engine.
 * @param image image data in BGR24 or Gray format.
 * @param width the width of image.
 * @param height the height of image.
 * @param pitch the byte offset between two lines of image.
 * @param bpp 8: gray image, 24: BGR image.
 * @param eyes the coordinates of two eyes.
 * @param feature extracted feature template.
 * @return if success return 0, else return the error code.
 */
typedef int (*TAimManualEnroll)(HAIM handle, const unsigned char* image, int width, int height, int pitch, int bpp,
	const POINT* eyes, unsigned char* feature);

extern TAimCreate			IAimCreate;
extern TAimFree				IAimFree;
extern TAimSetParams		IAimSetParams;
extern TAimProcess			IAimProcess;
extern TAimMatch			IAimMatch;
extern TAimManualEnroll		IAimManualEnroll;

HMODULE LoadAPI(const char* fileName);
void FreeAPI(HMODULE handle);

#endif