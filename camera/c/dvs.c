#include <libcaer/libcaer.h>
#include <libcaer/devices/samsung_evk.h>

#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if !defined(__APPLE__)
#include <unistd.h>
#endif

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

static uint16_t g_width = 0;
static uint16_t g_height = 0;

static inline uint32_t IDX(uint16_t x, uint16_t y) {
    return ((uint32_t)y * (uint32_t)g_width) + (uint32_t)x;
}

typedef struct {
    uint16_t *accum_u16;
    uint8_t *image_u8;

    uint32_t last_event_count;
    uint32_t tick_count;

    float gain;
    float decay;
    float dt_s;
} BaselineState;

BaselineState *baseline = NULL;

void initialize(uint16_t w, uint16_t h) {
    g_width = w;
    g_height = h;
}

static void baselineAlloc(BaselineState *b) {
    const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;
    b->accum_u16 = (uint16_t *)calloc(N, sizeof(uint16_t));
    b->image_u8 = (uint8_t *)calloc(N, sizeof(uint8_t));
}

static void baselineFree(BaselineState *b) {
    free(b->accum_u16);
    free(b->image_u8);
}

void initializeBaseline(float gain, float decay) {
    if (baseline != NULL) {
        return;
    }

    baseline = (BaselineState *)calloc(1, sizeof(BaselineState));
    baselineAlloc(baseline);

    baseline->gain = gain;
    if (!isfinite(baseline->gain) || baseline->gain < 0.0f) {
        baseline->gain = 0.0f;
    }

    baseline->decay = decay;
    if (!isfinite(baseline->decay) || baseline->decay < 0.0f) {
        baseline->decay = 0.0f;
    }
    if (baseline->decay > 1.0f) {
        baseline->decay = 1.0f;
    }

    baseline->dt_s = 0.05f;
}

void stopBaseline() {
    if (baseline == NULL) {
        return;
    }
    baselineFree(baseline);
    free(baseline);
    baseline = NULL;
}

void baselineSetParams(float gain, float decay) {
    if (baseline == NULL) {
        return;
    }

    baseline->gain = gain;
    if (!isfinite(baseline->gain) || baseline->gain < 0.0f) {
        baseline->gain = 0.0f;
    }

    baseline->decay = decay;
    if (!isfinite(baseline->decay) || baseline->decay < 0.0f) {
        baseline->decay = 0.0f;
    }
    if (baseline->decay > 1.0f) {
        baseline->decay = 1.0f;
    }
}

static inline void accumEvent(BaselineState *b, uint16_t x, uint16_t y, bool p) {
    (void)p;
    if (x >= g_width || y >= g_height) {
        return;
    }
    uint32_t idx = IDX(x, y);
    uint16_t v = b->accum_u16[idx];
    if (v != UINT16_MAX) {
        b->accum_u16[idx] = (uint16_t)(v + 1);
    }
}

void processEvents(uint32_t count, int32_t *timestamps, uint16_t *x, uint16_t *y, bool *p) {
    (void)timestamps;
    if (baseline == NULL) {
        return;
    }

    baseline->last_event_count = count;
    for (uint32_t i = 0; i < count; i++) {
        accumEvent(baseline, x[i], y[i], p[i]);
    }
}

static void buildImage(BaselineState *b) {
    const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;

    for (uint32_t i = 0; i < N; i++) {
        float scaled = (float)b->accum_u16[i] * b->gain;
        if (scaled > 255.0f) {
            scaled = 255.0f;
        }
        b->image_u8[i] = (uint8_t)(scaled + 0.5f);
    }
}

static void applyDecay(BaselineState *b) {
    const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;
    float d = b->decay;

    if (d <= 0.0f) {
        memset(b->accum_u16, 0, N * sizeof(uint16_t));
        return;
    }
    if (d >= 1.0f) {
        return;
    }

    for (uint32_t i = 0; i < N; i++) {
        b->accum_u16[i] = (uint16_t)((float)b->accum_u16[i] * d);
    }
}

void baselineTick(float dt_s) {
    if (baseline == NULL) {
        return;
    }

    baseline->dt_s = dt_s;
    baseline->tick_count++;

    buildImage(baseline);
    applyDecay(baseline);
}

struct sigaction shutdownAction;
caerDeviceHandle handle = NULL;
static atomic_bool globalShutdown = ATOMIC_VAR_INIT(false);

static void globalShutdownSignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        atomic_store(&globalShutdown, true);
    }
}

static void usbShutdownHandler(void *ptr) {
    (void)(ptr);
    atomic_store(&globalShutdown, true);
}

int initialize_camera() {
    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags = 0;
    sigemptyset(&shutdownAction.sa_mask);
    sigaddset(&shutdownAction.sa_mask, SIGTERM);
    sigaddset(&shutdownAction.sa_mask, SIGINT);

    if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
        caerLog(CAER_LOG_CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGTERM. Error:%d.", errno);
        return (EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
        caerLog(CAER_LOG_CRITICAL, "ShutdownAction",
                "Failed to set signal handler for SIGINT. Error:%d.", errno);
        return (EXIT_FAILURE);
    }

    handle = caerDeviceOpen(1, CAER_DEVICE_SAMSUNG_EVK, 0, 0, NULL);
    if (handle == NULL) {
        return (EXIT_FAILURE);
    }

    struct caer_samsung_evk_info info = caerSamsungEVKInfoGet(handle);
    printf("%s -- ID: %d, DVS X: %d, DVS Y: %d.\n",
           info.deviceString, info.deviceID, info.dvsSizeX, info.dvsSizeY);

    caerDeviceSendDefaultConfig(handle);
    return (EXIT_SUCCESS);
}

void start_camera() {
    caerDeviceDataStart(handle, NULL, NULL, NULL, &usbShutdownHandler, NULL);

    caerDeviceConfigSet(handle, CAER_HOST_CONFIG_DATAEXCHANGE,
                        CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, 0);
}

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
            continue;
        }

        if (caerEventPacketHeaderGetEventType(packetHeader) == POLARITY_EVENT) {
            caerPolarityEventPacket polarity = (caerPolarityEventPacket)packetHeader;
            int32_t count = caerEventPacketHeaderGetEventNumber(&polarity->packetHeader);
            if (count <= 0) {
                continue;
            }

            if (baseline != NULL) {
                baseline->last_event_count = (uint32_t)count;
            }

            for (int32_t j = 0; j < count; j++) {
                caerPolarityEventConst e = caerPolarityEventPacketGetEventConst(polarity, j);
                uint16_t x = caerPolarityEventGetX(e);
                uint16_t y = caerPolarityEventGetY(e);
                bool p = caerPolarityEventGetPolarity(e);

                if (baseline != NULL) {
                    accumEvent(baseline, x, y, p);
                }
            }

            eventCount += count;
        }
    }

    caerEventPacketContainerFree(packetContainer);
    return eventCount;
}

void shutdown() {
    if (baseline != NULL) {
        stopBaseline();
    }
    if (handle != NULL) {
        stop_camera();
        close_camera();
    }
}
