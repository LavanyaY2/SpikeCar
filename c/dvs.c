#include <libcaer/libcaer.h>
#include <libcaer/devices/samsung_evk.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>

int32_t currentTime = 0;
uint16_t width = 0;
uint16_t height = 0;



typedef struct {
	float_t *magnitude;
	float_t tau;
	float_t scale;
} timesurface;


timesurface *ts = NULL;






void timesurfaceInit(timesurface *ts, float_t tau) {
	ts->magnitude = calloc(width*height, sizeof(float_t));
	ts->tau = tau * 1e6;
	ts->scale = 1/tau;
}

void timesurfaceFree(timesurface *ts) {
	free(ts->magnitude);
}


void timesurfaceProcessEvent(timesurface* ts, int32_t timestamp, uint16_t x, uint16_t y, bool p) {	
	int32_t index = y*width + x;
	float_t power = (timestamp-currentTime)/(ts->tau);
	if (power>2) power=2;
	float_t scale = ts->scale * exp(power);
	if (!p) {
		scale = -scale;
	}
    ts->magnitude[index] += scale;
}

void timesurfaceUpdateTime(timesurface* ts, int32_t now) {
	float_t power = (now-currentTime)/(ts->tau);
	if (power>2) power=2;
	float_t scale = exp(-power);
	
	int32_t limit = width*height;
	for (int i=0; i<limit; i++) {
		ts->magnitude[i]*=scale;
		if (!isfinite(ts->magnitude[i])) {
		    ts->magnitude[i] = FLT_MAX;
		}
	}
}




void initialize(uint16_t w, uint16_t h) {
	width = w;
	height = h;
}

void initializeTimesurface(float_t timesurfaceTau) {
	ts = calloc(1, sizeof(timesurface));
	timesurfaceInit(ts, timesurfaceTau);
}
void stopTimesurface() {
	if (ts==NULL) return;
	timesurfaceFree(ts);
	free(ts);
	ts = NULL;
}



void processEvent(int32_t timestamp, uint16_t x, uint16_t y, bool p) {
	if (ts!=NULL) timesurfaceProcessEvent(ts, timestamp, x, y, p);
}

void processEvents(uint32_t count, int32_t* timestamps, uint16_t* x, uint16_t* y,
		   bool* p) {
	for (uint32_t i=0; i < count; i++) {
		processEvent(timestamps[i], x[i], y[i], p[i]);
	}
	if (ts!=NULL) timesurfaceUpdateTime(ts, timestamps[count-1]);
	currentTime = timestamps[count-1];
}


/***************************************************
 * shutdown handler based on
 * https://github.com/inivation/libcaer/blob/master/examples/davis_simple.c
 *
 */
struct sigaction shutdownAction;
caerDeviceHandle handle = NULL;
static atomic_bool globalShutdown = ATOMIC_VAR_INIT(false);

static void globalShutdownSignalHandler(int signal) {
	if (signal == SIGTERM || signal == SIGINT) {
		atomic_store(&globalShutdown, true);
	}
}
static void usbShutdownHandler(void *ptr) {
	(void) (ptr); // UNUSED.
	atomic_store(&globalShutdown, true);
}

/*************************************************************************
 * Initialize interface to DVX, based on
 * https://github.com/inivation/libcaer/blob/master/examples/davis_simple.c
 */
int initialize_camera() {
	shutdownAction.sa_handler = &globalShutdownSignalHandler;
	shutdownAction.sa_flags   = 0;
	sigemptyset(&shutdownAction.sa_mask);
	sigaddset(&shutdownAction.sa_mask, SIGTERM);
	sigaddset(&shutdownAction.sa_mask, SIGINT);

	if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
		caerLog(CAER_LOG_CRITICAL, "ShutdownAction", 
		        "Failed to set signal handler for SIGTERM. Error:%d.",
		       	errno);
		return (EXIT_FAILURE);
	}

	if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
		caerLog(CAER_LOG_CRITICAL, "ShutdownAction", 
			"Failed to set signal handler for SIGINT. Error:%d.",
		        errno);
		return (EXIT_FAILURE);
	}


	// Open a DVX, give it a device ID of 1,
	// and don't care about USB bus or SN restrictions.
	handle = caerDeviceOpen(1, CAER_DEVICE_SAMSUNG_EVK, 0, 0, NULL);
	if (handle == NULL) {
		return (EXIT_FAILURE);
	}

	// Let's take a look at the information we have on the device.
	struct caer_samsung_evk_info info = caerSamsungEVKInfoGet(handle);
	printf("%s -- ID: %d, DVS X: %d, DVS Y: %d.\n",
	       info.deviceString,
	       info.deviceID,
	       info.dvsSizeX,
	       info.dvsSizeY);

	// Send the default configuration before using the device.
	// No configuration is sent automatically!
	caerDeviceSendDefaultConfig(handle);

	return (EXIT_SUCCESS);
}

/*************************************************************************
 * Start gathering data from the camera
 */
void start_camera(bool blocking) {
    caerDeviceDataStart(handle, NULL, NULL, NULL,
		        &usbShutdownHandler, NULL);

    // Set blocking mode
    caerDeviceConfigSet(handle, CAER_HOST_CONFIG_DATAEXCHANGE,
		   	CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, blocking);
    
}

/*************************************************************************
 * Stop gathering data from the camera
 */
void stop_camera() {
	caerDeviceDataStop(handle);
}

void close_camera() {
	caerDeviceClose(&handle);
}

int update_camera() {
	caerEventPacketContainer packetContainer = caerDeviceDataGet(handle);
	if (packetContainer == NULL) {
		return -1;
	}

	int32_t packetNum = caerEventPacketContainerGetEventPacketsNumber(packetContainer);
	int eventCount = 0;

	for (int32_t i = 0; i < packetNum; i++) {
		caerEventPacketHeader packetHeader = caerEventPacketContainerGetEventPacket(packetContainer, i);
		if (packetHeader == NULL) {
			continue; // Skip if nothing there.
		}
		if (i == POLARITY_EVENT) {
			caerPolarityEventPacket polarity = (caerPolarityEventPacket) packetHeader;
			int32_t count = caerEventPacketHeaderGetEventNumber(&polarity->packetHeader);
			int32_t last_timestamp = 0;
			for (int32_t j = 0; j < count; j++) {
				caerPolarityEventConst event = caerPolarityEventPacketGetEventConst(polarity, j);
				int32_t ts = caerPolarityEventGetTimestamp(event);
				uint16_t x = caerPolarityEventGetX(event);
				uint16_t y = caerPolarityEventGetY(event);
				bool p = caerPolarityEventGetPolarity(event);

				processEvent(ts, x, y, p);

				eventCount ++;
				last_timestamp = ts;

			}
			if (count > 0) {
	            if (ts!=NULL) timesurfaceUpdateTime(ts, last_timestamp);
	            currentTime = last_timestamp;
			
			}
		} 
	}
	caerEventPacketContainerFree(packetContainer);
	
	
	return eventCount;
}





void shutdown() {
    if (ts!=NULL) stopTimesurface();

	if (handle!=NULL) {
	    stop_camera();
	    close_camera();
	}
}

