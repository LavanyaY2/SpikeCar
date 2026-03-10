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

// -------------------- Global sensor dims --------------------
static uint16_t g_width = 0;
static uint16_t g_height = 0;

static inline uint32_t IDX(uint16_t x, uint16_t y)
{
	return ((uint32_t)y * (uint32_t)g_width) + (uint32_t)x;
}

// -------------------- Tunables / Limits --------------------
#define MAX_BLOBS 64
#define MAX_TRACKS 32
#define IOU_MERGE_THRESH 0.20f
#define MERGE_GAP_PX 6

typedef struct
{
	uint16_t id;
	uint32_t area;
	uint16_t xmin, ymin, xmax, ymax;
	float cx, cy;
} Blob;

typedef struct
{
	uint16_t id;
	uint8_t active;

	// curr centroid position of the object
	float cx, cy;
	// current area
	float A;
	// previous area
	float A_prev;

	// instantaneous derivative
	float A_dot;
	// smoothed derivative: uses Exponential Moving Average (EMA)
	float A_dot_ema;
	// seconds, or INFINITY if not approaching
	float ttc;

	// how long has this track existed
	uint16_t age;
	// time it wasn't able to match it's blob
	uint16_t missed;

	// check if in ROI
	uint8_t in_roi;
	// count consecutive ticks
	uint8_t growth_streak;
} Track;

// Struct that owns all the buffers and results
typedef struct
{
	// Buffers

	// this is the raw accumulator of all the events; camera dims H*W
	uint16_t *accum_u16;
	// scaled image; i.e. a normalized human representation of events
	// this is an 8 bit intensity map basically
	uint8_t *image_u8;
	// this is the mask to apply on accum_u16 to get image_u8 H*W (0/255)
	uint8_t *mask_u8;
	// this tells us the blobs and labels
	// so similar pixels will come together to form blob and get an id
	uint16_t *labels_u16;

	// metadata

	// number of DVS events recently stored
	uint32_t last_event_count;
	// how many times processing loop has run
	uint32_t tick_count;

	// how many blobs have been seen
	uint16_t blob_count;
	// tracks all blobs seen so far
	Blob blobs[MAX_BLOBS];
	// tracks blobs as they go through
	Track tracks[MAX_TRACKS];

	// tracks what is the most critical track at the moment
	int16_t critical_track_id; // -1 if none
	float critical_ttc;

	// Parameters for frame (moved from python to C)
	// this is the accum->image scale
	// set to 50 for best results (too low and it's too dark)
	float gain;
	// 0..1 (0 clears accum each tick) -> how long should events be relevant for
	float decay;
	// image_u8 threshold for mask
	uint8_t thresh;
	// minimum blob area - this helps remove a lot of noise
	uint32_t area_min;

	// assumed tick dt in seconds (passed by Python)
	// used to compute derivative
	float dt_s;
	// this is used to match tracks with blobs
	float match_radius;

	// TTC smoothing for EMA
	float ema_alpha;	   // 0..1?
	uint8_t growth_needed; // is this growing to be critical or no?

	// ROI rectangle coords
	uint16_t roi_x0, roi_y0, roi_x1, roi_y1;

	// internal track id counter
	uint16_t next_track_id;
} BaselineState;

// Exported this global pointer so Python can use dll("baseline")
BaselineState *baseline = NULL;

/**
 * @brief initialize by setting camera dimensions
 *
 * @param w Width of camera
 * @param h Height of camera
 */
void initialize(uint16_t w, uint16_t h)
{
	g_width = w;
	g_height = h;
}

/**
 * @brief Allocate buffer ptrs for baseline obj
 *
 * @param b
 */
static void baselineAlloc(BaselineState *b)
{
	const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;
	b->accum_u16 = (uint16_t *)calloc(N, sizeof(uint16_t));
	b->image_u8 = (uint8_t *)calloc(N, sizeof(uint8_t));
	b->mask_u8 = (uint8_t *)calloc(N, sizeof(uint8_t));
	b->labels_u16 = (uint16_t *)calloc(N, sizeof(uint16_t));
}

/**
 * @brief free up buffer ptrs for baseline obj.
 *
 * @param b
 */
static void baselineFree(BaselineState *b)
{
	free(b->accum_u16);
	free(b->image_u8);
	free(b->mask_u8);
	free(b->labels_u16);
}

static void tracksReset(BaselineState *b)
{
	for (int i = 0; i < MAX_TRACKS; i++)
	{
		b->tracks[i].active = 0;
		b->tracks[i].id = 0;
		b->tracks[i].cx = 0;
		b->tracks[i].cy = 0;
		b->tracks[i].A = 0;
		b->tracks[i].A_prev = 0;
		b->tracks[i].A_dot = 0;
		b->tracks[i].A_dot_ema = 0;
		b->tracks[i].ttc = INFINITY;
		b->tracks[i].age = 0;
		b->tracks[i].missed = 0;
		b->tracks[i].in_roi = 0;
		b->tracks[i].growth_streak = 0;
	}
	b->next_track_id = 1;
	b->critical_track_id = -1;
	b->critical_ttc = INFINITY;
}

/**
 * @brief initialize baseline
 *
 * following params are provided from command line
 *
 * current workign version
 * python3 view.py --gain 20 --tick_ms 100 --area_min 220 --growth_needed 4 --ema_alpha 0.25 --roi_x0 0 --roi_y0 0 --roi_x1 639 --roi_y1 479
 * @param gain
 * @param decay
 * @param thresh
 * @param area_min
 * @param match_radius
 * @param ema_alpha
 * @param growth_needed
 * @param roi_x0
 * @param roi_y0
 * @param roi_x1
 * @param roi_y1
 */
void initializeBaseline(
	float gain,
	float decay,
	uint8_t thresh,
	uint32_t area_min,
	float match_radius,
	float ema_alpha,
	uint8_t growth_needed,
	uint16_t roi_x0, uint16_t roi_y0, uint16_t roi_x1, uint16_t roi_y1)
{
	if (baseline != NULL)
		return;

	baseline = (BaselineState *)calloc(1, sizeof(BaselineState));
	baselineAlloc(baseline);

	baseline->gain = gain;
	baseline->decay = decay;
	if (baseline->decay < 0.0f)
		baseline->decay = 0.0f;
	if (baseline->decay > 1.0f)
		baseline->decay = 1.0f;

	baseline->thresh = thresh;
	baseline->area_min = area_min;

	baseline->match_radius = match_radius;

	baseline->ema_alpha = ema_alpha;
	if (baseline->ema_alpha < 0.0f)
		baseline->ema_alpha = 0.0f;
	if (baseline->ema_alpha > 1.0f)
		baseline->ema_alpha = 1.0f;

	baseline->growth_needed = growth_needed;

	baseline->roi_x0 = roi_x0;
	baseline->roi_y0 = roi_y0;
	baseline->roi_x1 = roi_x1;
	baseline->roi_y1 = roi_y1;

	baseline->dt_s = 0.05f; // default, overwritten by tick(dt_s)
	tracksReset(baseline);
}

void stopBaseline()
{
	if (baseline == NULL)
		return;
	baselineFree(baseline);
	free(baseline);
	baseline = NULL;
}

void baselineSetParams(float gain, float decay, uint8_t thresh, uint32_t area_min)
{
	if (baseline == NULL)
		return;
	baseline->gain = gain;
	baseline->decay = decay;
	if (baseline->decay < 0.0f)
		baseline->decay = 0.0f;
	if (baseline->decay > 1.0f)
		baseline->decay = 1.0f;

	baseline->thresh = thresh;
	baseline->area_min = area_min;
}

void baselineSetROI(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
	if (baseline == NULL)
		return;
	baseline->roi_x0 = x0;
	baseline->roi_y0 = y0;
	baseline->roi_x1 = x1;
	baseline->roi_y1 = y1;
}

// -------------------- Event accumulation (timestamps ignored) --------------------
static inline void accumEvent(BaselineState *b, uint16_t x, uint16_t y, bool p)
{
	(void)p;
	if (x >= g_width || y >= g_height)
		return;
	uint32_t idx = IDX(x, y);
	uint16_t v = b->accum_u16[idx];
	if (v != UINT16_MAX)
		b->accum_u16[idx] = (uint16_t)(v + 1);
}

void processEvents(uint32_t count, int32_t *timestamps, uint16_t *x, uint16_t *y, bool *p)
{
	(void)timestamps;
	if (baseline == NULL)
		return;

	baseline->last_event_count = count;
	for (uint32_t i = 0; i < count; i++)
	{
		accumEvent(baseline, x[i], y[i], p[i]);
	}
}

// -------------------- Build image + mask + labels + blobs --------------------

// Convert accum -> image_u8, build mask, clear labels
static void buildImages(BaselineState *b)
{
	const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;

	float gain = b->gain;
	if (!isfinite(gain) || gain < 0.0f)
		gain = 0.0f;

	uint8_t thr = b->thresh;

	for (uint32_t i = 0; i < N; i++)
	{
		float scaled = (float)b->accum_u16[i] * gain;
		if (scaled > 255.0f)
			scaled = 255.0f;
		uint8_t pix = (uint8_t)(scaled + 0.5f);
		b->image_u8[i] = pix;
		b->mask_u8[i] = (pix >= thr) ? 255 : 0;
		b->labels_u16[i] = 0;
	}
}

// Connected components (4-connectivity) with a simple flood fill using a queue.
// This is not the fastest possible, but it is robust and easy to debug.
typedef struct
{
	uint16_t x, y;
} QNode;

static void findBlobs(BaselineState *b)
{
	b->blob_count = 0;

	const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;

	// One queue reused for all floods (max size N)
	// Note: for 640x480, this is big but ok on PC. On embedded you’d optimize.
	QNode *queue = (QNode *)malloc(N * sizeof(QNode));
	if (queue == NULL)
		return;

	uint16_t next_label = 1;

	for (uint16_t y = 0; y < g_height; y++)
	{
		for (uint16_t x = 0; x < g_width; x++)
		{
			uint32_t idx = IDX(x, y);
			if (b->mask_u8[idx] == 0)
				continue;
			if (b->labels_u16[idx] != 0)
				continue;

			if (b->blob_count >= MAX_BLOBS)
			{
				// Still label it to avoid re-processing, but don’t store metadata
				// Flood-fill labeling only
			}

			// Flood fill
			uint32_t qh = 0, qt = 0;
			queue[qt++] = (QNode){x, y};
			b->labels_u16[idx] = next_label;

			uint32_t area = 0;
			uint32_t sumx = 0, sumy = 0;
			uint16_t xmin = x, xmax = x, ymin = y, ymax = y;

			while (qh < qt)
			{
				QNode n = queue[qh++];
				uint16_t nx = n.x, ny = n.y;

				area++;
				sumx += nx;
				sumy += ny;
				if (nx < xmin)
					xmin = nx;
				if (nx > xmax)
					xmax = nx;
				if (ny < ymin)
					ymin = ny;
				if (ny > ymax)
					ymax = ny;

				// 4-neighbors
				if (nx > 0)
				{
					uint16_t xx = nx - 1;
					uint32_t ii = IDX(xx, ny);
					if (b->mask_u8[ii] && b->labels_u16[ii] == 0)
					{
						b->labels_u16[ii] = next_label;
						queue[qt++] = (QNode){xx, ny};
					}
				}
				if (nx + 1 < g_width)
				{
					uint16_t xx = nx + 1;
					uint32_t ii = IDX(xx, ny);
					if (b->mask_u8[ii] && b->labels_u16[ii] == 0)
					{
						b->labels_u16[ii] = next_label;
						queue[qt++] = (QNode){xx, ny};
					}
				}
				if (ny > 0)
				{
					uint16_t yy = ny - 1;
					uint32_t ii = IDX(nx, yy);
					if (b->mask_u8[ii] && b->labels_u16[ii] == 0)
					{
						b->labels_u16[ii] = next_label;
						queue[qt++] = (QNode){nx, yy};
					}
				}
				if (ny + 1 < g_height)
				{
					uint16_t yy = ny + 1;
					uint32_t ii = IDX(nx, yy);
					if (b->mask_u8[ii] && b->labels_u16[ii] == 0)
					{
						b->labels_u16[ii] = next_label;
						queue[qt++] = (QNode){nx, yy};
					}
				}
			}

			// Store blob if big enough
			if (area >= b->area_min && b->blob_count < MAX_BLOBS)
			{
				Blob *bl = &b->blobs[b->blob_count++];
				bl->id = next_label;
				bl->area = area;
				bl->xmin = xmin;
				bl->xmax = xmax;
				bl->ymin = ymin;
				bl->ymax = ymax;
				bl->cx = (area > 0) ? ((float)sumx / (float)area) : (float)x;
				bl->cy = (area > 0) ? ((float)sumy / (float)area) : (float)y;
			}

			next_label++;
		}
	}

	free(queue);
}

static inline uint16_t u16min(uint16_t a, uint16_t b) { return (a < b) ? a : b; }
static inline uint16_t u16max(uint16_t a, uint16_t b) { return (a > b) ? a : b; }

static inline uint32_t boxArea(const Blob *bl)
{
	return (uint32_t)(bl->xmax - bl->xmin + 1) * (uint32_t)(bl->ymax - bl->ymin + 1);
}

static inline float blobIoU(const Blob *a, const Blob *b)
{
	uint16_t ix0 = u16max(a->xmin, b->xmin);
	uint16_t iy0 = u16max(a->ymin, b->ymin);
	uint16_t ix1 = u16min(a->xmax, b->xmax);
	uint16_t iy1 = u16min(a->ymax, b->ymax);

	if (ix1 < ix0 || iy1 < iy0)
		return 0.0f;

	uint32_t iw = (uint32_t)(ix1 - ix0 + 1);
	uint32_t ih = (uint32_t)(iy1 - iy0 + 1);
	uint32_t inter = iw * ih;
	uint32_t ua = boxArea(a);
	uint32_t ub = boxArea(b);
	uint32_t uni = ua + ub - inter;
	if (uni == 0)
		return 0.0f;
	return (float)inter / (float)uni;
}

static inline uint8_t boxesNear(const Blob *a, const Blob *b, int gap_px)
{
	int ax0 = (int)a->xmin - gap_px;
	int ay0 = (int)a->ymin - gap_px;
	int ax1 = (int)a->xmax + gap_px;
	int ay1 = (int)a->ymax + gap_px;

	return !(ax1 < (int)b->xmin || ax0 > (int)b->xmax || ay1 < (int)b->ymin || ay0 > (int)b->ymax);
}

static void mergeBlobInto(Blob *dst, const Blob *src)
{
	uint32_t a1 = dst->area;
	uint32_t a2 = src->area;
	uint32_t at = a1 + a2;
	if (at == 0)
		at = 1;

	dst->xmin = u16min(dst->xmin, src->xmin);
	dst->ymin = u16min(dst->ymin, src->ymin);
	dst->xmax = u16max(dst->xmax, src->xmax);
	dst->ymax = u16max(dst->ymax, src->ymax);
	dst->cx = ((dst->cx * (float)a1) + (src->cx * (float)a2)) / (float)at;
	dst->cy = ((dst->cy * (float)a1) + (src->cy * (float)a2)) / (float)at;
	dst->area = at;
}

static void mergeBlobsIoU(BaselineState *b)
{
	if (b->blob_count <= 1)
		return;

	Blob merged[MAX_BLOBS];
	uint8_t consumed[MAX_BLOBS];
	memset(consumed, 0, sizeof(consumed));

	uint16_t out_count = 0;
	for (uint16_t i = 0; i < b->blob_count && out_count < MAX_BLOBS; i++)
	{
		if (consumed[i])
			continue;

		Blob cur = b->blobs[i];
		consumed[i] = 1;

		uint8_t changed = 1;
		while (changed)
		{
			changed = 0;
			for (uint16_t j = 0; j < b->blob_count; j++)
			{
				if (consumed[j])
					continue;

				float iou = blobIoU(&cur, &b->blobs[j]);
				if (iou >= IOU_MERGE_THRESH || boxesNear(&cur, &b->blobs[j], MERGE_GAP_PX))
				{
					mergeBlobInto(&cur, &b->blobs[j]);
					consumed[j] = 1;
					changed = 1;
				}
			}
		}

		cur.id = (uint16_t)(out_count + 1);
		merged[out_count++] = cur;
	}

	for (uint16_t k = 0; k < out_count; k++)
		b->blobs[k] = merged[k];
	b->blob_count = out_count;
}

static inline uint8_t pointInROI(BaselineState *b, float cx, float cy)
{
	return (cx >= b->roi_x0 && cx <= b->roi_x1 && cy >= b->roi_y0 && cy <= b->roi_y1) ? 1 : 0;
}

static void reduceToFinalObstacle(BaselineState *b)
{
	if (b->blob_count <= 1)
		return;

	int best = -1;
	uint32_t best_area = 0;

	for (uint16_t i = 0; i < b->blob_count; i++)
	{
		Blob *bl = &b->blobs[i];
		uint8_t in_roi = pointInROI(b, bl->cx, bl->cy);
		if (!in_roi)
			continue;
		if (bl->area > best_area)
		{
			best = (int)i;
			best_area = bl->area;
		}
	}

	// Fallback when no merged blob is currently in ROI.
	if (best < 0)
	{
		for (uint16_t i = 0; i < b->blob_count; i++)
		{
			if (b->blobs[i].area > best_area)
			{
				best = (int)i;
				best_area = b->blobs[i].area;
			}
		}
	}

	if (best >= 0)
	{
		Blob chosen = b->blobs[best];
		chosen.id = 1;
		b->blobs[0] = chosen;
		b->blob_count = 1;
	}
}

// -------------------- Tracking + TTC --------------------
static int findFreeTrackSlot(BaselineState *b)
{
	for (int i = 0; i < MAX_TRACKS; i++)
	{
		if (!b->tracks[i].active)
			return i;
	}
	return -1;
}

static int matchTrack(BaselineState *b, const Blob *bl, uint8_t *track_used)
{
	float best_d2 = INFINITY;
	int best_i = -1;

	float r = b->match_radius;
	float r2 = r * r;

	for (int i = 0; i < MAX_TRACKS; i++)
	{
		if (!b->tracks[i].active)
			continue;
		if (track_used[i])
			continue;

		float dx = b->tracks[i].cx - bl->cx;
		float dy = b->tracks[i].cy - bl->cy;
		float d2 = dx * dx + dy * dy;
		if (d2 < best_d2)
		{
			best_d2 = d2;
			best_i = i;
		}
	}

	if (best_i >= 0 && best_d2 <= r2)
		return best_i;
	return -1;
}

static void updateTracks(BaselineState *b)
{
	// Mark all tracks as unmatched initially
	uint8_t used[MAX_TRACKS];
	memset(used, 0, sizeof(used));

	// Age/missed update will happen after matching
	// We’ll store which tracks got updated
	uint8_t updated[MAX_TRACKS];
	memset(updated, 0, sizeof(updated));

	// Match blobs to tracks
	for (uint16_t bi = 0; bi < b->blob_count; bi++)
	{
		const Blob *bl = &b->blobs[bi];

		int ti = matchTrack(b, bl, used);
		if (ti < 0)
		{
			// Create new track if possible
			int free_i = findFreeTrackSlot(b);
			if (free_i >= 0)
			{
				Track *tr = &b->tracks[free_i];
				tr->active = 1;
				tr->id = b->next_track_id++;
				tr->cx = bl->cx;
				tr->cy = bl->cy;
				tr->A = (float)bl->area;
				tr->A_prev = (float)bl->area;
				tr->A_dot = 0.0f;
				tr->A_dot_ema = 0.0f;
				tr->ttc = INFINITY;
				tr->age = 1;
				tr->missed = 0;
				tr->in_roi = pointInROI(b, tr->cx, tr->cy);
				tr->growth_streak = 0;

				used[free_i] = 1;
				updated[free_i] = 1;
			}
		}
		else
		{
			// Update existing track
			Track *tr = &b->tracks[ti];
			used[ti] = 1;
			updated[ti] = 1;

			tr->missed = 0;
			tr->age++;

			tr->cx = bl->cx;
			tr->cy = bl->cy;

			tr->A_prev = tr->A;
			tr->A = (float)bl->area;

			// derivative using assumed dt
			float dt = b->dt_s;
			if (dt <= 1e-6f)
				dt = 1e-6f;

			tr->A_dot = (tr->A - tr->A_prev) / dt;
			tr->A_dot_ema = b->ema_alpha * tr->A_dot + (1.0f - b->ema_alpha) * tr->A_dot_ema;

			// Growth streak
			if (tr->A_dot_ema > 0.0f)
				tr->growth_streak++;
			else
				tr->growth_streak = 0;

			tr->in_roi = pointInROI(b, tr->cx, tr->cy);

			// TTC
			if (tr->A_dot_ema > 1e-3f)
				tr->ttc = tr->A / tr->A_dot_ema;
			else
				tr->ttc = INFINITY;
		}
	}

	// Any track not updated gets missed++
	for (int i = 0; i < MAX_TRACKS; i++)
	{
		if (!b->tracks[i].active)
			continue;
		if (updated[i])
			continue;

		// Single-obstacle pipeline: stale tracks can create false choices in decision logic.
		// Drop unmatched tracks immediately.
		b->tracks[i].active = 0;
	}
}

// Select critical track with gating
static void selectCritical(BaselineState *b)
{
	b->critical_track_id = -1;
	b->critical_ttc = INFINITY;

	for (int i = 0; i < MAX_TRACKS; i++)
	{
		Track *tr = &b->tracks[i];
		if (!tr->active)
			continue;

		// Gating
		if (!tr->in_roi)
			continue;
		if (tr->growth_streak < b->growth_needed)
			continue;
		if (tr->age < 2)
			continue;
		if (!isfinite(tr->ttc))
			continue;
		if (tr->ttc <= 0.0f)
			continue;

		if (tr->ttc < b->critical_ttc)
		{
			b->critical_ttc = tr->ttc;
			b->critical_track_id = (int16_t)tr->id;
		}
	}
}

// Apply accumulator decay after tick
static void applyDecay(BaselineState *b)
{
	const uint32_t N = (uint32_t)g_width * (uint32_t)g_height;
	float d = b->decay;

	if (d <= 0.0f)
	{
		memset(b->accum_u16, 0, N * sizeof(uint16_t));
		return;
	}
	if (d >= 1.0f)
		return;

	for (uint32_t i = 0; i < N; i++)
	{
		b->accum_u16[i] = (uint16_t)((float)b->accum_u16[i] * d);
	}
}

// Called from Python once per UI frame/tick.
// NO timestamps involved. dt_s is whatever your UI loop uses.
// TODO: Time this algorithm
void baselineTick(float dt_s)
{
	if (baseline == NULL)
		return;

	baseline->dt_s = dt_s;
	baseline->tick_count++;

	buildImages(baseline);
	findBlobs(baseline);
	mergeBlobsIoU(baseline);
	reduceToFinalObstacle(baseline);
	updateTracks(baseline);
	selectCritical(baseline);
	applyDecay(baseline);
}

// -------------------- Camera boilerplate (libcaer) --------------------
struct sigaction shutdownAction;
caerDeviceHandle handle = NULL;
static atomic_bool globalShutdown = ATOMIC_VAR_INIT(false);

static void globalShutdownSignalHandler(int signal)
{
	if (signal == SIGTERM || signal == SIGINT)
	{
		atomic_store(&globalShutdown, true);
	}
}
static void usbShutdownHandler(void *ptr)
{
	(void)(ptr);
	atomic_store(&globalShutdown, true);
}

int initialize_camera()
{
	shutdownAction.sa_handler = &globalShutdownSignalHandler;
	shutdownAction.sa_flags = 0;
	sigemptyset(&shutdownAction.sa_mask);
	sigaddset(&shutdownAction.sa_mask, SIGTERM);
	sigaddset(&shutdownAction.sa_mask, SIGINT);

	if (sigaction(SIGTERM, &shutdownAction, NULL) == -1)
	{
		caerLog(CAER_LOG_CRITICAL, "ShutdownAction",
				"Failed to set signal handler for SIGTERM. Error:%d.", errno);
		return (EXIT_FAILURE);
	}
	if (sigaction(SIGINT, &shutdownAction, NULL) == -1)
	{
		caerLog(CAER_LOG_CRITICAL, "ShutdownAction",
				"Failed to set signal handler for SIGINT. Error:%d.", errno);
		return (EXIT_FAILURE);
	}

	handle = caerDeviceOpen(1, CAER_DEVICE_SAMSUNG_EVK, 0, 0, NULL);
	if (handle == NULL)
	{
		return (EXIT_FAILURE);
	}

	struct caer_samsung_evk_info info = caerSamsungEVKInfoGet(handle);
	printf("%s -- ID: %d, DVS X: %d, DVS Y: %d.\n",
		   info.deviceString, info.deviceID, info.dvsSizeX, info.dvsSizeY);

	caerDeviceSendDefaultConfig(handle);
	return (EXIT_SUCCESS);
}

// Python calls start_camera() with no args -> must match signature
void start_camera()
{
	caerDeviceDataStart(handle, NULL, NULL, NULL, &usbShutdownHandler, NULL);

	// Non-blocking so UI stays responsive
	caerDeviceConfigSet(handle, CAER_HOST_CONFIG_DATAEXCHANGE,
						CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, 0);
}

void stop_camera() { caerDeviceDataStop(handle); }
void close_camera() { caerDeviceClose(&handle); }

// Pull one container of events and accumulate into baseline->accum_u16.
// NO baselineTick() here—Python controls the tick rate (dt).
int update_camera()
{
	caerEventPacketContainer packetContainer = caerDeviceDataGet(handle);
	if (packetContainer == NULL)
		return -1;

	int32_t packetNum = caerEventPacketContainerGetEventPacketsNumber(packetContainer);
	int eventCount = 0;

	for (int32_t i = 0; i < packetNum; i++)
	{
		caerEventPacketHeader packetHeader = caerEventPacketContainerGetEventPacket(packetContainer, i);
		if (packetHeader == NULL)
			continue;

		if (caerEventPacketHeaderGetEventType(packetHeader) == POLARITY_EVENT)
		{
			caerPolarityEventPacket polarity = (caerPolarityEventPacket)packetHeader;
			int32_t count = caerEventPacketHeaderGetEventNumber(&polarity->packetHeader);
			if (count <= 0)
				continue;

			if (baseline != NULL)
				baseline->last_event_count = (uint32_t)count;

			for (int32_t j = 0; j < count; j++)
			{
				caerPolarityEventConst e = caerPolarityEventPacketGetEventConst(polarity, j);
				uint16_t x = caerPolarityEventGetX(e);
				uint16_t y = caerPolarityEventGetY(e);
				bool p = caerPolarityEventGetPolarity(e);

				if (baseline != NULL)
					accumEvent(baseline, x, y, p);
			}

			eventCount += count;
		}
	}

	caerEventPacketContainerFree(packetContainer);
	return eventCount;
}

void shutdown()
{
	if (baseline != NULL)
		stopBaseline();
	if (handle != NULL)
	{
		stop_camera();
		close_camera();
	}
}

float baselineGetCriticalTTC()
{
	if (baseline == NULL)
		return INFINITY;
	return baseline->critical_ttc;
}

uint32_t baselineGetLastEventCount()
{
	if (baseline == NULL)
		return 0;
	return baseline->last_event_count;
}