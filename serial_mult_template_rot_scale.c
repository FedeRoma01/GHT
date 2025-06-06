// generalized_hough_serial_multitemplate_multiscale_rotation.c

// ROTAZIONE CAMBIATA
// DETECTION CERCHIO ROSSO + BORDO VERDE
// NMS ==> Risulato basatu su questo e risulato in overlay su scena

/*
diagramma visuale per rappresentare il flusso combinato di **scale** e **rotazioni** nel codice multitemplate:
Per ogni TEMPLATE:
└─ per ogni SCALA (0.8x, 1.0x, 1.2x):
    └─ ridimensiona il template → resized[]
    └─ per ogni ANGOLO (0°, 30°, ..., 330°):
        └─ ruota il template ridimensionato → rotated[]
        └─ calcola gradienti e bordi → tgrad_x[], tgrad_y[], tedges[]
        └─ costruisce lookup table per (scala, angolo)
        └─ esegue Generalized Hough Transform sulla scena
        └─ salva l’accumulatore → file: accumulator_t[TEMPLATE]_s[SCALA]_a[ANGOLO].pgm
**Flusso visualizzato:**
- A ogni livello (template → scala → angolo) si generano combinazioni indipendenti.
- Ogni combinazione produce un risultato separato, utile per verificare dove e come è stato trovato il match migliore.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#include <time.h>

#define ANGLE_BINS 360
#define GRADIENT_THRESHOLD 100
#define NUM_SCALES 5
#define NUM_ANGLES 12 // (30°)
#define MAX_O_X_BIN 100

typedef struct {
    int dx, dy;
} Offset;

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
    memset(magnitude, 0, width * height * sizeof(float));
    memset(grad_x, 0, width * height * sizeof(float)); //
    memset(grad_y, 0, width * height * sizeof(float)); //

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum_x = 0, sum_y = 0;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++) {
                    int xi = x + j;
                    int yi = y + i;
                    sum_x += gx[i+1][j+1] * img[yi * width + xi];
                    sum_y += gy[i+1][j+1] * img[yi * width + xi];
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

void build_lookup_table(unsigned char *edges, float *grad_x, float *grad_y, int width, int height, int *counter, int control) {
    memset(lookup_count, 0, sizeof(lookup_count));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {
                // COUNTER PIXEL DI BORDO
                if (control) (*counter)++;

                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                Offset o = { width / 2 - x, height / 2 - y };
                if (lookup_count[bin] < MAX_O_X_BIN) lookup_table[bin][lookup_count[bin]++] = o;
            }
        }
    }
    //if (control) printf("COUNTER: %d\n", *counter);
}

void save_bordi_pgm(const char *filename, unsigned char *bordi, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("Errore salvataggio PGM"); exit(EXIT_FAILURE); }
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        fwrite(&bordi[i], 1, 1, fp);
    }
    fclose(fp);
}

void generalized_hough(unsigned char *edges, float *grad_x, float *grad_y, int *accumulator, int width, int height, int *max) {
    int counter2 = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {
                counter2++;
                //bordi_votanti[idx] = 255;
                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                for (int b = -1; b <= 1; b++) {
                    int bidx = (bin + b + ANGLE_BINS) % ANGLE_BINS;
                    for (int i = 0; i < lookup_count[bidx]; i++) {
                        int xc = x + lookup_table[bidx][i].dx;
                        int yc = y + lookup_table[bidx][i].dy;
                        if (xc >= 0 && xc < width && yc >= 0 && yc < height) {
                            accumulator[yc * width + xc]++;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < width * height; i++) {
        if (accumulator[i] > *max) *max = accumulator[i];
    }
    printf("GH PIXEL MASSIMO: %d\n", *max);
    //printf("PIXEL TOTALI CHE VOTANO: %d\n", counter2);
}

void save_profile(const char *filename, const char *field, double time) {
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file");
        return;
    }
    fprintf(fp, "%s %f\n", field, time);
    fclose(fp);
}

void save_nms_result_pgm(const char *filename, int *nms_res, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("Errore salvataggio PGM"); exit(EXIT_FAILURE); }
    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    int max_vote = 1;
    unsigned char black = 0;
    for (int i = 0; i < width * height; i++) {
        if (nms_res[i] > max_vote) max_vote = nms_res[i];
    }
    printf("VOTO MASSIMO %s: %d\n", filename, max_vote);
    for (int i = 0; i < width * height; i++) {
        unsigned char val = (unsigned char)(255.0 * nms_res[i] / max_vote);
        if (val >= 240) {      /////////////////////////////// possibile causa di problemi
            fwrite(&val, 1, 1, fp);
            int y = i / width;
            int x = i % width;
            //printf("CENTRO RILEVATO: X = %d, Y = %d\n", x, y);
        } else fwrite(&black, 1, 1, fp);
    }
    fclose(fp);
}

unsigned char* rotate_image_bilinear_expand(unsigned char *src, int width, int height, float angle_degrees, int *new_width, int *new_height) {
    float angle_radians = angle_degrees * (M_PI / 180.0f);
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

            int x0 = (int)floor(src_x);
            int x1 = x0 + 1;
            int y0 = (int)floor(src_y);
            int y1 = y0 + 1;

            if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height) {
                float dx = src_x - x0;
                float dy = src_y - y0;

                unsigned char p00 = src[y0 * width + x0];
                unsigned char p10 = src[y0 * width + x1];
                unsigned char p01 = src[y1 * width + x0];
                unsigned char p11 = src[y1 * width + x1];

                float w00 = (1 - dx) * (1 - dy);
                float w10 = dx * (1 - dy);
                float w01 = (1 - dx) * dy;
                float w11 = dx * dy;

                float value = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;
                if (value < 0) value = 0;
                if (value > 255) value = 255;

                dst[y * (*new_width) + x] = (unsigned char)(value + 0.5f);
            }
        }
    }

    return dst;
}

void save_edges_pgm(const char *filename, unsigned char *edges, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot open file for writing");
        return;
    }

    // Scrivo l'intestazione PGM (P5 indica formato binario)
    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    // Scrivo i dati dell'immagine (pixel)
    size_t written = fwrite(edges, sizeof(unsigned char), width * height, fp);
    if (written != width * height) {
        fprintf(stderr, "Warning: wrote only %zu bytes (expected %d)\n", written, width * height);
    }

    fclose(fp);
}


void non_maximum_suppression(int *accumulator, int *nms_result, int width, int height, int window_size) {
    int half = window_size / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int current = accumulator[y * width + x];
            int is_max = 1;

            for (int wy = -half; wy <= half; wy++) {
                for (int wx = -half; wx <= half; wx++) {
                    int nx = x + wx;
                    int ny = y + wy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        if (accumulator[ny * width + nx] > current) {
                            is_max = 0;
                            goto fine_controllo;
                        }
                    }
                }
            }
        fine_controllo:
            nms_result[y * width + x] = is_max ? current : 0;
        }
    }
}

void save_detection_overlay(const char *filename, unsigned char *scene, int *nms_res, int width, int height, int tw_s, int th_s, int max, int scene_max) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("Errore salvataggio PPM"); exit(EXIT_FAILURE); }

    fprintf(fp, "P6\n%d %d\n255\n", width, height); // P6 = RGB binary

    printf("MASSIMO nella scena: %d\n", scene_max);
    int threshold = (int)(0.95 * (scene_max));

    printf("THRESHOLD: %d (%d)\n", threshold, max);
    int circle_radius = 5;

    // Prepara immagine RGB (3 byte per pixel)
    unsigned char *output = malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        output[3 * i + 0] = scene[i]; // R
        output[3 * i + 1] = scene[i]; // G
        output[3 * i + 2] = scene[i]; // B
    }

    // Per ogni centro rilevato: disegna cerchio rosso + rettangolo verde
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (nms_res[idx] >= threshold) {
                printf("(%d %d) => %d\n", idx % width, idx / width, nms_res[idx]);
                // 1. Cerchio rosso
                for (int dy = -circle_radius; dy <= circle_radius; dy++) {
                    for (int dx = -circle_radius; dx <= circle_radius; dx++) {
                        int cx = x + dx;
                        int cy = y + dy;
                        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                            if (sqrt(dx * dx + dy * dy) <= circle_radius) {
                                int cidx = (cy * width + cx) * 3;
                                output[cidx + 0] = 255; // R
                                output[cidx + 1] = 0;   // G
                                output[cidx + 2] = 0;   // B
                            }
                        }
                    }
                }

                // 2. Rettangolo verde (bordi)
                int half_w = tw_s / 2;
                int half_h = th_s / 2;

                for (int dx = -half_w; dx <= half_w; dx++) {
                    int top_x = x + dx, top_y = y - half_h;
                    int bot_x = x + dx, bot_y = y + half_h;

                    if (top_x >= 0 && top_x < width && top_y >= 0 && top_y < height) {
                        int tidx = (top_y * width + top_x) * 3;
                        output[tidx + 0] = 0;   // R
                        output[tidx + 1] = 255; // G
                        output[tidx + 2] = 0;   // B
                    }
                    if (bot_x >= 0 && bot_x < width && bot_y >= 0 && bot_y < height) {
                        int bidx = (bot_y * width + bot_x) * 3;
                        output[bidx + 0] = 0;
                        output[bidx + 1] = 255;
                        output[bidx + 2] = 0;
                    }
                }

                for (int dy = -half_h; dy <= half_h; dy++) {
                    int left_x = x - half_w, left_y = y + dy;
                    int right_x = x + half_w, right_y = y + dy;

                    if (left_x >= 0 && left_x < width && left_y >= 0 && left_y < height) {
                        int lidx = (left_y * width + left_x) * 3;
                        output[lidx + 0] = 0;
                        output[lidx + 1] = 255;
                        output[lidx + 2] = 0;
                    }
                    if (right_x >= 0 && right_x < width && right_y >= 0 && right_y < height) {
                        int ridx = (right_y * width + right_x) * 3;
                        output[ridx + 0] = 0;
                        output[ridx + 1] = 255;
                        output[ridx + 2] = 0;
                    }
                }
            }
        }
    }

    // Scrive immagine RGB su file
    fwrite(output, 1, width * height * 3, fp);
    fclose(fp);
    free(output);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <template1.pgm> [template2.pgm ...] <scene.pgm>\n", argv[0]);
        return 1;
    }

    int num_templates = argc - 2;
    const char *scene_file = argv[argc - 1];

    int scene_w, scene_h;

    // SCENE LOADING
    unsigned char *scene_img = load_image_dynamic(scene_file, &scene_w, &scene_h);

    float *grad_x = malloc(scene_w * scene_h * sizeof(float));
    float *grad_y = malloc(scene_w * scene_h * sizeof(float));
    float *magnitude = malloc(scene_w * scene_h * sizeof(float));
    unsigned char *edges = malloc(scene_w * scene_h);
    int *accumulator = calloc(scene_w * scene_h, sizeof(int));
    int *nms_result = calloc(scene_w * scene_h, sizeof(int));

    // SCENE GRADIENT COMPUTATION
    compute_gradient(scene_img, grad_x, grad_y, magnitude, scene_w, scene_h);

    // SCENE EDGES COMPUTATION
    detect_edges(magnitude, edges, scene_w, scene_h);

    save_edges_pgm("scene_edges.pgm", edges, scene_w, scene_h);
 
    float scales[NUM_SCALES] = {0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    float angles[NUM_ANGLES];
    for (int a = 0; a < NUM_ANGLES; a++) angles[a] = a * (360.0 / NUM_ANGLES);

    for (int t = 1; t <= num_templates; t++) {
        int tw, th;
        int *counter = calloc(NUM_SCALES, sizeof(int));

        // TEMPLATE LOADING
        unsigned char *template_img = load_image_dynamic(argv[t], &tw, &th);
        //printf("DIM TEMPLATE: X = %d, Y = %d\n", tw, th);

        // TEMPLATE GRADIENT COMPUTATION
        float *tgrad_x = malloc(tw * th * sizeof(float));
        float *tgrad_y = malloc(tw * th * sizeof(float));
        float *tmagnitude = malloc(tw * th * sizeof(float));
        unsigned char *t_edges = malloc(tw * th);

        compute_gradient(template_img, tgrad_x, tgrad_y, tmagnitude, tw, th);   
        detect_edges(tmagnitude, t_edges, tw, th);
        
        free(tgrad_x);
        free(tgrad_y);
        free(tmagnitude);

        for (int si = 0; si < NUM_SCALES; si++) {
            int control = 1;
            printf("\nSCALA %.2f\n", scales[si]);
            for (int ai = 0; ai < NUM_ANGLES; ai++) {
                printf("ROTAZIONE %.2f\n", angles[ai]);
                int tw_s = (int)(tw * scales[si]);
                int th_s = (int)(th * scales[si]);
                unsigned char *resized = malloc(tw_s * th_s);
                unsigned char *rotated = malloc(tw_s * th_s);

                // TEMPLATE SCALING
                stbir_resize_uint8(t_edges, tw, th, 0, resized, tw_s, th_s, 0, 1);

                int new_w, new_h;
                // TEMPLATE ROTATION
                rotated = rotate_image_bilinear_expand(resized, tw_s, th_s, angles[ai], &new_w, &new_h);

                float *tgrad_x = malloc(new_w * new_h * sizeof(float));
                float *tgrad_y = malloc(new_w * new_h * sizeof(float));
                float *tmagnitude = malloc(new_w * new_h * sizeof(float));
                unsigned char *tedges = malloc(new_w * new_h);

                // TEMPLATE GRADIENT COMPUTATION
                compute_gradient(rotated, tgrad_x, tgrad_y, tmagnitude, new_w, new_h);

                // TEMPLATE EDGES COMPUTATION
                detect_edges(tmagnitude, tedges, new_w, new_h);

                char name[128];
                snprintf(name, sizeof(name), "edges_t%d_s%.1f_a%.0f.pgm", t, scales[si], angles[ai]);
             
                // TEMPLATE EDGES SAVING
                if (angles[ai] == 0.0f && scales[si] == 1.0f) save_edges_pgm(name, tedges, new_w, new_h);
                
                // TEMPLATE LOOKUP TABLE CREATION
                build_lookup_table(tedges, tgrad_x, tgrad_y, new_w, new_h, &counter[si], control);
                control = 0;

                int *taccumulator = calloc(new_w * new_h, sizeof(int));

                int max = 0;
                generalized_hough(tedges, tgrad_x, tgrad_y, taccumulator, new_w, new_h, &max);

                free(taccumulator);

                memset(accumulator, 0, scene_w * scene_h * sizeof(int));

                int scene_max = 0;
                generalized_hough(edges, grad_x, grad_y, accumulator, scene_w, scene_h, &scene_max);

                int window_size = (int)(fmax(new_w, new_h) / 4.0);
                if (window_size % 2 == 0) window_size++; // forza a dispari per avere pixel a dx e sx uguali
                //printf("WINDOW SIZE NMS: %d\n", window_size);

                non_maximum_suppression(accumulator, nms_result, scene_w, scene_h, window_size);

                char fname[128];
                snprintf(fname, sizeof(fname), "overlay_result_t%d_s%.1f_a%.0f.ppm", t, scales[si], angles[ai]);

                // TEMPLATE ACCUMULATOR SAVING
                save_detection_overlay(fname, scene_img, nms_result, scene_w, scene_h, new_w, new_h, max, scene_max);

                free(resized); free(rotated);
                free(tgrad_x); free(tgrad_y); free(tmagnitude); free(tedges);
            }
        }
        free(template_img);
        free(counter);
        free(t_edges);
    }

    free(scene_img); free(grad_x); free(grad_y); free(magnitude);
    free(edges); free(accumulator); free(nms_result);

    return 0;
}
