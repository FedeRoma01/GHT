#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ANGLE_BINS 360
#define DP 2
#define MIN_DISTANCE 80
#define GRADIENT_THRESHOLD 100
#define VOTE_THRESHOLD 300
#define NUM_ANGLES 4
#define MAX_O_X_BIN 100

typedef struct {
    int dx, dy;
} Offset;

typedef struct {
    int x;
    int y;
} Point;

Offset lookup_table[ANGLE_BINS][MAX_O_X_BIN];
int lookup_count[ANGLE_BINS] = {0};

unsigned char* load_image_dynamic(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { perror("Errore apertura file"); exit(EXIT_FAILURE); }
    char magic[3]; int maxval;
    fscanf(fp, "%2s", magic);
    if (strcmp(magic, "P5") != 0) { printf("Solo immagini P5 supportate\n"); exit(EXIT_FAILURE); }
    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", &maxval);
    fgetc(fp);
    unsigned char *img = malloc((*width) * (*height));
    fread(img, sizeof(unsigned char), (*width) * (*height), fp);
    fclose(fp);
    return img;
}

void compute_gradient(unsigned char *img, float *grad_x, float *grad_y, float *magnitude, int width, int height) {
    int gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int value;

    memset(magnitude, 0, width * height * sizeof(float));
    memset(grad_x, 0, width * height * sizeof(float)); //
    memset(grad_y, 0, width * height * sizeof(float)); //

    for (int y = 1; y < height - 1; y++) {
        for (int x = 0; x < width; x++) {
            float sum_x = 0, sum_y = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int xi = x + j;
                    int yi = y + i;
                    if (xi == -1 || xi == width)      // control for the right and left bounds 
                        value = img[yi * width + x];
                    else
                        value = img[yi * width + xi];
                    sum_x += gx[i+1][j+1] * value;
                    sum_y += gy[i+1][j+1] * value;
                }
            }
            int idx = y * width + x;
            grad_x[idx] = sum_x;
            grad_y[idx] = sum_y;
            magnitude[idx] = sqrt(sum_x * sum_x + sum_y * sum_y);
        }
    }
}

void detect_edges(float *magnitude, unsigned char *edges, int width, int height) {
    for (int i = 0; i < width * height; i++)
        edges[i] = (magnitude[i] > GRADIENT_THRESHOLD) ? 255 : 0;
}

void build_lookup_table(unsigned char *edges, float *grad_x, float *grad_y, int width, int height) {
    memset(lookup_count, 0, sizeof(lookup_count));
    memset(lookup_table, 0, sizeof(lookup_table));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {

                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                Offset o = { width / 2 - x, height / 2 - y };
                if (lookup_count[bin] < MAX_O_X_BIN) lookup_table[bin][lookup_count[bin]++] = o;
            }
        }
    }
}

void generalized_hough(unsigned char *edges, float *grad_x, float *grad_y, int width, int height, Point **finalDetections, int *detectionsCounter) {
    // Downscaling accumulator
    int acc_w = width / DP;
    int acc_h = height / DP;
    int *local_accumulator = calloc(acc_w * acc_h, sizeof(int));

    Point *detections = calloc(width * height, sizeof(Point)); // max one Point per cell

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {
                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                for (int i = 0; i < lookup_count[bin]; i++) {
                    int xc = x + lookup_table[bin][i].dx;
                    int yc = y + lookup_table[bin][i].dy;
                    int dx = xc / DP;
                    int dy = yc / DP;
                    if (dx >= 0 && dx < acc_w && dy >= 0 && dy < acc_h)
                        local_accumulator[dy * acc_w + dx]++;
                }
            }
        }
    }

    int num_det = 0;
    for (int y = 0; y < acc_h; y++) {
        for (int x = 0; x < acc_w; x++) {
            if (local_accumulator[y * acc_w + x] > VOTE_THRESHOLD) {
                detections[num_det].x = x * DP;
                detections[num_det].y = y * DP;
                num_det++;
            }
        }
    }

    // Non-Maximum Suppression
    *finalDetections = calloc(num_det, sizeof(Point));
    int finalCount = 0;

    for (int i = 0; i < num_det; i++) {
        int isMax = 1;
        for (int j = 0; j < num_det; j++) {
            if (i == j) continue;

            int dx = detections[i].x - detections[j].x;
            int dy = detections[i].y - detections[j].y;
            float distance = sqrtf(dx * dx + dy * dy);

            if (distance < MIN_DISTANCE) {
                int acc_i = local_accumulator[(detections[i].y / DP) * acc_w + (detections[i].x / DP)];
                int acc_j = local_accumulator[(detections[j].y / DP) * acc_w + (detections[j].x / DP)];
                if (acc_i < acc_j) {
                    isMax = 0;
                    break;
                }
            }
        }
        if (isMax) {
            (*finalDetections)[finalCount++] = detections[i];
        }
    }
    *detectionsCounter = finalCount; 

    free(local_accumulator);
    free(detections);
}

unsigned char* rotate_image_nearest_neighbor_expand(unsigned char *src, int width, int height, float angle_degrees, int *new_width, int *new_height) {
    float angle_radians = -angle_degrees * (M_PI / 180.0f);
    float cos_theta = cos(angle_radians);
    float sin_theta = sin(angle_radians);

    // Calcolo bounding box ruotata
    float corners_x[4] = { -width / 2.0f,  width / 2.0f,  width / 2.0f, -width / 2.0f };
    float corners_y[4] = { -height / 2.0f, -height / 2.0f, height / 2.0f,  height / 2.0f };

    float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;

    for (int i = 0; i < 4; i++) {
        float x_rot = cos_theta * corners_x[i] - sin_theta * corners_y[i];
        float y_rot = sin_theta * corners_x[i] + cos_theta * corners_y[i];
        if (x_rot < min_x) min_x = x_rot;
        if (x_rot > max_x) max_x = x_rot;
        if (y_rot < min_y) min_y = y_rot;
        if (y_rot > max_y) max_y = y_rot;
    }

    *new_width  = (int)(ceil(max_x - min_x));
    *new_height = (int)(ceil(max_y - min_y));

    unsigned char *dst = calloc((*new_width) * (*new_height), sizeof(unsigned char)); // fondo nero

    int cx_src = width / 2;
    int cy_src = height / 2;
    int cx_dst = *new_width / 2;
    int cy_dst = *new_height / 2;

    // Offset di riallineamento per mantenere il centro originale
    float offset_x = cx_dst - (cos_theta * cx_src - sin_theta * cy_src);
    float offset_y = cy_dst - (sin_theta * cx_src + cos_theta * cy_src);

    for (int y = 0; y < *new_height; y++) {
        for (int x = 0; x < *new_width; x++) {
            float xt = x - offset_x;
            float yt = y - offset_y;

            float src_x =  cos_theta * xt + sin_theta * yt;
            float src_y = -sin_theta * xt + cos_theta * yt;

            int nearest_x = (int)(roundf(src_x));
            int nearest_y = (int)(roundf(src_y));

            if (nearest_x >= 0 && nearest_x < width && nearest_y >= 0 && nearest_y < height) {
                dst[y * (*new_width) + x] = src[nearest_y * width + nearest_x];
            }
        }
    }

    return dst;
}

void draw_circle(unsigned char *image, int img_w, int img_h, Point center, int radius, unsigned char r, unsigned char g, unsigned char b) {
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int x = center.x + dx;
            int y = center.y + dy;
            if (x >= 0 && x < img_w && y >= 0 && y < img_h && dx*dx + dy*dy <= radius*radius) {
                int idx = (y * img_w + x) * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
            }
        }
    }
}

void draw_rectangle(unsigned char *image, int img_w, int img_h, int x, int y, int w, int h, unsigned char r, unsigned char g, unsigned char b) {
    for (int i = 0; i < w; i++) {
        int xt = x + i;
        if (xt >= 0 && xt < img_w) {
            if (y >= 0 && y < img_h) {
                int top_idx = (y * img_w + xt) * 3;
                image[top_idx + 0] = r;
                image[top_idx + 1] = g;
                image[top_idx + 2] = b;
            }
            if ((y + h) >= 0 && (y + h) < img_h) {
                int bot_idx = ((y + h) * img_w + xt) * 3;
                image[bot_idx + 0] = r;
                image[bot_idx + 1] = g;
                image[bot_idx + 2] = b;
            }
        }
    }
    for (int i = 0; i < h; i++) {
        int yt = y + i;
        if (yt >= 0 && yt < img_h) {
            if (x >= 0 && x < img_w) {
                int left_idx = (yt * img_w + x) * 3;
                image[left_idx + 0] = r;
                image[left_idx + 1] = g;
                image[left_idx + 2] = b;
            }
            if ((x + w) >= 0 && (x + w) < img_w) {
                int right_idx = (yt * img_w + (x + w)) * 3;
                image[right_idx + 0] = r;
                image[right_idx + 1] = g;
                image[right_idx + 2] = b;
            }
        }
    }
}


void save_edges_pgm(const char *filename, unsigned char *edges, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot open file for writing");
        return;
    }

    // PGM heading (P5 binary format)
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    // Write image pixels
    size_t written = fwrite(edges, sizeof(unsigned char), width * height, fp);
    if (written != width * height) {
        fprintf(stderr, "Warning: wrote only %zu bytes (expected %d)\n", written, width * height);
    }

    fclose(fp);
}

int main(int argc, char **argv) {
    int rank, size;

    int scene_w = 0, scene_h = 0;
    unsigned char *scene_img = load_image_dynamic("resources/scene_key.pgm", &scene_w, &scene_h);
    int scene_size = scene_w * scene_h;

    float *grad_x = malloc(scene_size * sizeof(float));
    float *grad_y = malloc(scene_size * sizeof(float));
    float *magnitude = malloc(scene_size * sizeof(float));
    unsigned char *edges = malloc(scene_size);

    compute_gradient(scene_img, grad_x, grad_y, magnitude, scene_w, scene_h);
    detect_edges(magnitude, edges, scene_w, scene_h);

    save_edges_pgm("scene_edges.pgm", edges, scene_w, scene_h);

    float angles[NUM_ANGLES];
    for (int a = 0; a < NUM_ANGLES; a++) angles[a] = a * (360.0 / NUM_ANGLES);

    int tw = 0, th = 0;

    // TEMPLATE LOADING
    unsigned char *template_img = load_image_dynamic("resources/templ_key.pgm", &tw, &th);

    for (int ai = 0; ai < NUM_ANGLES; ai++) {
        int x = tw;
        int y = th;
        unsigned char *rotated = rotate_image_nearest_neighbor_expand(template_img, x, y, angles[ai], &tw, &th);

        float *tgrad_x = malloc(tw * th * sizeof(float));
        float *tgrad_y = malloc(tw * th * sizeof(float));
        float *tmagnitude = malloc(tw * th * sizeof(float));
        unsigned char *tedges = malloc(tw * th * sizeof(unsigned char));

        compute_gradient(rotated, tgrad_x, tgrad_y, tmagnitude, tw, th);
        detect_edges(tmagnitude, tedges, tw, th);
        build_lookup_table(tedges, tgrad_x, tgrad_y, tw, th);

        // edge template saving (decomment if needed)
        //char name[128];
        //snprintf(name, sizeof(name), "edges_a%.0f.pgm", angles[ai]);
        //save_edges_pgm(name, tedges, tw, th);

        free(rotated);
        free(tgrad_x); free(tgrad_y); free(tmagnitude); free(tedges);

        Point *finalDetections = NULL;
        int detectionsCounter = 0;
        generalized_hough(edges, grad_x, grad_y, scene_w, scene_h, &finalDetections, &detectionsCounter);

        if (detectionsCounter > 0) {
        
            char fname[128];
            snprintf(fname, sizeof(fname), "result_a%.0f.ppm", angles[ai]);
            printf("Num detection (%s): %d\n", fname, detectionsCounter);

            unsigned char *scene_rgb = malloc(scene_w * scene_h * 3);
            for (int i = 0; i < scene_w * scene_h; i++) {
                scene_rgb[3*i + 0] = scene_img[i];  // R
                scene_rgb[3*i + 1] = scene_img[i];  // G
                scene_rgb[3*i + 2] = scene_img[i];  // B
            }

            for (int i = 0; i < detectionsCounter; i++) {
                Point center = finalDetections[i];

                // Draw circle in blue (RGB: 255, 0, 0)
                draw_circle(scene_rgb, scene_w, scene_h, center, 10, 255, 0, 0);

                // Draw bounding box in green (RGB: 0, 255, 0)
                int box_x = center.x - tw / 2;
                int box_y = center.y - th / 2;
                draw_rectangle(scene_rgb, scene_w, scene_h, box_x, box_y, tw, th, 0, 255, 0);

                //printf("Draw circle in (%d,%d) with radius %d\n", center.x, center.y, 10);
                //printf("Draw rectangle in (%d,%d) with dimensions %dx%d\n", box_x, box_y, tw, th);
            }

            // Optional: save or write image
            FILE *fp = fopen(fname, "wb");
            if (!fp) {
                perror("Errore apertura file di output");
                continue;
            }
            fprintf(fp, "P6\n%d %d\n255\n", scene_w, scene_h);
            fwrite(scene_rgb, 1, scene_w * scene_h * 3, fp);
            fclose(fp);

            free(scene_rgb);
        }
        free(finalDetections);
    }

    free(template_img);

    free(scene_img); free(grad_x); free(grad_y); free(magnitude);
    free(edges);

    return 0;
}
