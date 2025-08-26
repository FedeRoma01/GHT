// Batch per process
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/types.h>
#include <dirent.h>

#define ANGLE_BINS 360
#define DP 2
#define MIN_DISTANCE 80
#define GRADIENT_THRESHOLD 100
#define VOTE_THRESHOLD 300
#define NUM_ANGLES 4
#define MAX_O_X_BIN 100
#define MAX_FILENAME_LEN 256
#define MAX_FILES 512

typedef struct {
    int dx, dy;
} Offset;

typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    Offset table[ANGLE_BINS][MAX_O_X_BIN]; // 2D array: bin → offset list
    int count[ANGLE_BINS];                 // number of offset for each bin
    float angle;                           // rotation (degree)
    int tw;
    int th;                           
} LookupTable;

LookupTable *lookup_tables;

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

void compute_gradient(unsigned char *img, float *grad_x, float *grad_y, float *magnitude, int width, int height, int flag) {
    int gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int value;

    if (flag) {
        memset(magnitude, 0, width * height * sizeof(float));
        memset(grad_x, 0, width * height * sizeof(float)); //
        memset(grad_y, 0, width * height * sizeof(float)); //
    }

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

void compute_gradient_mpi(unsigned char *global_img, float *global_grad_x, float *global_grad_y, float *global_magnitude, int width, int height, MPI_Comm comm, FILE *profile_fp) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double timestamp_start, timestamp_end;

    int rows_per_proc = height / size;
    int extra = height % size;

    int local_height = rows_per_proc + (rank < extra ? 1 : 0);
    int start_row = rank * rows_per_proc + (rank < extra ? rank : extra);

    // // Add 2 rows for halo
    int buffer_height = local_height + 2;
    unsigned char *local_img = calloc(buffer_height * width, sizeof(unsigned char));
    float *local_grad_x = calloc(buffer_height * width, sizeof(float));
    float *local_grad_y = calloc(buffer_height * width, sizeof(float));
    float *local_magnitude = calloc(buffer_height * width, sizeof(float));

    // Manual Scatterv (more flexible)
    int *sendcounts = calloc(size, sizeof(int));
    int *displs = calloc(size, sizeof(int));
    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < extra ? 1 : 0);
        sendcounts[i] = rows * width;
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    timestamp_start = MPI_Wtime();
    MPI_Scatterv(global_img, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 &local_img[width], local_height * width, MPI_UNSIGNED_CHAR,
                 0, comm);
    timestamp_end = MPI_Wtime();
    fprintf(profile_fp, "MPI_Scatterv (CG): %.6f s\n", timestamp_end - timestamp_start);

    timestamp_start = MPI_Wtime();
    // Halo exchange
    if (rank > 0) {
        MPI_Sendrecv(&local_img[width], width, MPI_UNSIGNED_CHAR, rank-1, 0,
                     &local_img[0], width, MPI_UNSIGNED_CHAR, rank-1, 1,
                     comm, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Sendrecv(&local_img[local_height * width], width, MPI_UNSIGNED_CHAR, rank+1, 1,
                     &local_img[(local_height + 1) * width], width, MPI_UNSIGNED_CHAR, rank+1, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    timestamp_end = MPI_Wtime();
    fprintf(profile_fp, "MPI_Sendrecv (CG): %.6f s\n", timestamp_end - timestamp_start);

    // Computation on the assigned block (halo rows considered only for boundary rows)
    compute_gradient(local_img, local_grad_x, local_grad_y, local_magnitude, width, buffer_height, 0);

    timestamp_start = MPI_Wtime();
    // Reduce to the real block
    MPI_Gatherv(&local_grad_x[width], local_height * width, MPI_FLOAT,
                global_grad_x, sendcounts, displs, MPI_FLOAT, 0, comm);
    MPI_Gatherv(&local_grad_y[width], local_height * width, MPI_FLOAT,
                global_grad_y, sendcounts, displs, MPI_FLOAT, 0, comm);
    MPI_Gatherv(&local_magnitude[width], local_height * width, MPI_FLOAT,
                global_magnitude, sendcounts, displs, MPI_FLOAT, 0, comm);
    timestamp_end = MPI_Wtime();
    fprintf(profile_fp, "MPI_Gatherv (CG): %.6f s\n", timestamp_end - timestamp_start);

    free(local_img);
    free(local_grad_x);
    free(local_grad_y);
    free(local_magnitude);
    free(sendcounts);
    free(displs);
}

void detect_edges(float *magnitude, unsigned char *edges, int width, int height) {
    for (int i = 0; i < width * height; i++)
        edges[i] = (magnitude[i] > GRADIENT_THRESHOLD) ? 255 : 0;
}

void build_lookup_table(unsigned char *edges, float *grad_x, float *grad_y, int width, int height, LookupTable *lt) {
    memset(lt->count, 0, sizeof(lt->count));
    lt->tw = width;
    lt->th = height;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {
                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                Offset o = { width / 2 - x, height / 2 - y };
                int count = lt->count[bin];
                if (count < MAX_O_X_BIN) {
                    lt->table[bin][count] = o;
                    lt->count[bin]++;
                }
            }
        }
    }
}

void generalized_hough(unsigned char *edges, float *grad_x, float *grad_y, int width, int height, Point **finalDetections, int *detectionsCounter, int ai) {
    // Downscaling accumulator
    int acc_w = width / DP;
    int acc_h = height / DP;
    int *local_accumulator = calloc(acc_w * acc_h, sizeof(int));

    Point *detections = calloc(width * height, sizeof(Point)); // max one Point per cell

    LookupTable *lt = &lookup_tables[ai];

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (edges[idx] == 255) {
                float angle = atan2(grad_y[idx], grad_x[idx]);
                if (angle < 0) angle += 2 * M_PI;
                int bin = (int)(angle * (ANGLE_BINS / (2 * M_PI))) % ANGLE_BINS;
                for (int i = 0; i < lt->count[bin]; i++) {
                    int xc = x + lt->table[bin][i].dx;
                    int yc = y + lt->table[bin][i].dy;
                    int dx = xc / DP;
                    int dy = yc / DP;
                    if (dx >= 0 && dx < acc_w && dy >= 0 && dy < acc_h)
                        local_accumulator[dy * acc_w + dx]++;
                }
            }
        }
    }

    // Peak detection
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

    // Compute ruotated bounding box
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

    unsigned char *dst = calloc((*new_width) * (*new_height), sizeof(unsigned char)); // black blackground

    int cx_src = width / 2;
    int cy_src = height / 2;
    int cx_dst = *new_width / 2;
    int cy_dst = *new_height / 2;

    // Realignment offset to maintain the original center
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

const char* get_extension(const char *filename) {
    const char *dot = strrchr(filename, '.');
    return (!dot || dot == filename) ? "" : dot;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);

    double total_start = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double timestamp_start, timestamp_end;
    FILE *profile_fp = NULL;
    char profile_filename[64];
    snprintf(profile_filename, sizeof(profile_filename), "profiling_rank_%d.txt", rank);
    profile_fp = fopen(profile_filename, "w");
    if (!profile_fp) {
        fprintf(stderr, "Errore apertura file di profiling per rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int tw = 0, th = 0;
    lookup_tables = malloc(NUM_ANGLES * sizeof(LookupTable));

    // Template loading
    unsigned char *template_img = NULL;
    if (rank == 0) {
        timestamp_start = MPI_Wtime();
        template_img = load_image_dynamic("resources/templ_key.pgm", &tw, &th);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "load_image_dynamic: %.6f s\n", timestamp_end - timestamp_start);
    }

    float angles[NUM_ANGLES];
    for (int a = 0; a < NUM_ANGLES; a++) angles[a] = a * (360.0 / NUM_ANGLES);

    // Pre-processing look-up tables
    for (int a = 0; a < NUM_ANGLES; a++) {
        float angle = angles[a];
        unsigned char *rotated = NULL;

        lookup_tables[a].angle = angle;

        int tw_i, th_i = 0; // rotated img dim

        if (rank == 0) {
            timestamp_start = MPI_Wtime();
            rotated = rotate_image_nearest_neighbor_expand(template_img, tw, th, angle, &tw_i, &th_i);
            timestamp_end = MPI_Wtime();
            fprintf(profile_fp, "rotate_image_nearest_neighbor_expand: %.6f s\n", timestamp_end - timestamp_start);
        }

        int templ_i_dims[2];
        if (rank == 0) { templ_i_dims[0]= tw_i; templ_i_dims[1]= th_i; }
        timestamp_start = MPI_Wtime();
        MPI_Bcast(templ_i_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "MPI_Bcast: %.6f s\n", timestamp_end - timestamp_start);
        tw_i = templ_i_dims[0]; th_i = templ_i_dims[1];

        float *tgrad_x = malloc(tw_i * th_i * sizeof(float));
        float *tgrad_y = malloc(tw_i * th_i * sizeof(float));
        float *tmagnitude = malloc(tw_i * th_i * sizeof(float));
        unsigned char *tedges = malloc(tw_i * th_i * sizeof(unsigned char));

        // Template gradient computation parallelized
        timestamp_start = MPI_Wtime();
        compute_gradient_mpi(rotated, tgrad_x, tgrad_y, tmagnitude, tw_i, th_i, MPI_COMM_WORLD, profile_fp);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "compute_gradient (a=%.0f): %.6f s\n", angle, timestamp_end - timestamp_start);

        int scene_i_size = tw_i * th_i;

        // Bcast gradients for single generalized_hough

        // Creating the derived type for the three arrays
        MPI_Datatype tgradient_type;
        int blocklengths[3] = {scene_i_size, scene_i_size, scene_i_size};
        MPI_Aint displs[3];
        MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};

        // Calculating relative addresses
        MPI_Aint base_addr;
        MPI_Get_address(tgrad_x, &base_addr);
        MPI_Get_address(tgrad_x, &displs[0]);
        MPI_Get_address(tgrad_y, &displs[1]);
        MPI_Get_address(tmagnitude, &displs[2]);

        for (int i = 0; i < 3; i++) {
            displs[i] = displs[i] - base_addr;
        }

        MPI_Type_create_struct(3, blocklengths, displs, types, &tgradient_type);
        MPI_Type_commit(&tgradient_type);

        // Unique Broadcast
        MPI_Bcast(tgrad_x, 1, tgradient_type, 0, MPI_COMM_WORLD);

        MPI_Type_free(&tgradient_type);

        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "MPI_Bcast: %.6f s\n", timestamp_end - timestamp_start);

        // end Bcast of gradients


        // Template edge detection
        timestamp_start = MPI_Wtime();
        detect_edges(tmagnitude, tedges, tw_i, th_i);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "detect_edges (a=%.0f): %.6f s\n", angle, timestamp_end - timestamp_start);

        // Look-up table construction
        timestamp_start = MPI_Wtime();
        build_lookup_table(tedges, tgrad_x, tgrad_y, tw_i, th_i, &lookup_tables[a]);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "build_lookup_table (a=%.0f): %.6f s\n", angle, timestamp_end - timestamp_start);

        
        if (rank == 0) {
            char name[128];
            snprintf(name, sizeof(name), "edges_a%.0f.pgm", angles[a]);
            save_edges_pgm(name, tedges, tw_i, th_i);
        }

        free(rotated);
        free(tgrad_x); free(tgrad_y); free(tmagnitude); free(tedges);
    
    }

    free(template_img);

    // File list recovering
    int num_files = 0;
    char **file_list = NULL;
    if (rank == 0) {
        // Scenes distribution
        char *dir_path = malloc(MAX_FILENAME_LEN);
        snprintf(dir_path, MAX_FILENAME_LEN, "%s%s", "resources/dataset/", argv[1]);
        const char *extension = ".pgm"; // includi il punto: es. ".pgm"

        DIR *dir = opendir(dir_path);
        if (!dir) {
            perror("opendir");
            return EXIT_FAILURE;
        }

        file_list = malloc(MAX_FILES * sizeof(char *));

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && num_files < MAX_FILES) {
            if (entry->d_type == DT_REG) {
                if (strcmp(get_extension(entry->d_name), extension) == 0) {
                    char *path = malloc(MAX_FILENAME_LEN);
                    snprintf(path, MAX_FILENAME_LEN, "%s/%s", dir_path, entry->d_name);
                    file_list[num_files++] = path;
                }
            }
        }
        closedir(dir);

        printf("Trovati %d file '%s' in %s\n", num_files, extension, dir_path);
    }

    timestamp_start = MPI_Wtime();
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
    timestamp_end = MPI_Wtime();
    fprintf(profile_fp, "MPI_Bcast: %.6f s\n", timestamp_end - timestamp_start);

    // File distribution: each process receives the paths it needs to process
    int *send_counts = NULL;
    int *displs = NULL;
    char *all_paths_buffer = NULL; // contiguous string buffer
    int *path_lengths = NULL;

    if (rank == 0) {
        // Calculate string lengths and create contiguous buffers
        path_lengths = malloc(num_files * sizeof(int));
        int total_chars = 0;
        for (int i = 0; i < num_files; i++) {
            path_lengths[i] = strlen(file_list[i]) + 1; // +1 for terminator '\0'
            total_chars += path_lengths[i];
        }

        all_paths_buffer = malloc(total_chars);
        int offset = 0;
        for (int i = 0; i < num_files; i++) {
            memcpy(all_paths_buffer + offset, file_list[i], path_lengths[i]);
            offset += path_lengths[i];
        }

        // Distribution calculation: each process receives num_files / size (+ remainder)
        send_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int base = num_files / size;
        int rest = num_files % size;

        int file_index = 0;
        for (int r = 0; r < size; r++) {
            int files_for_rank = base + (r < rest ? 1 : 0);

            int chars_for_rank = 0;
            for (int f = 0; f < files_for_rank; f++) {
                chars_for_rank += path_lengths[file_index++];
            }

            send_counts[r] = chars_for_rank;
            displs[r] = (r == 0 ? 0 : displs[r-1] + send_counts[r-1]);
        }
    }

    timestamp_start = MPI_Wtime();
    // Each process receives the number of characters from the buffer it needs to read.
    int my_chars = 0;
    if (rank == 0) {
        MPI_Scatter(send_counts, 1, MPI_INT, &my_chars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, 1, MPI_INT, &my_chars, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    char *my_paths_buffer = malloc(my_chars);

    // Distributes the actual routes
    MPI_Scatterv(all_paths_buffer, send_counts, displs, MPI_CHAR,
                 my_paths_buffer, my_chars, MPI_CHAR, 0, MPI_COMM_WORLD);
    timestamp_end = MPI_Wtime();
    fprintf(profile_fp, "MPI_Scatter: %.6f s\n", timestamp_end - timestamp_start);

    char **my_files = NULL;
    int my_files_count = 0;
    {
        // Count how many strings in my buffer
        for (int i = 0; i < my_chars; i++) {
            if (my_paths_buffer[i] == '\0')
                my_files_count++;
        }

        my_files = malloc(my_files_count * sizeof(char *));
        int idx = 0;
        my_files[0] = my_paths_buffer;
        for (int i = 0; i < my_chars - 1; i++) {
            if (my_paths_buffer[i] == '\0') {
                my_files[++idx] = &my_paths_buffer[i+1];
            }
        }
    }

    // Each process load its own images locally
    for (int i = 0; i < my_files_count; i++) {

        // Scene loading
        int scene_w = 0, scene_h = 0;
        unsigned char *scene_img = load_image_dynamic(my_files[i], &scene_w, &scene_h);
        int scene_size = scene_w * scene_h;

        float *grad_x = malloc(scene_size * sizeof(float));
        float *grad_y = malloc(scene_size * sizeof(float));
        float *magnitude = malloc(scene_size * sizeof(float));
        unsigned char *edges = malloc(scene_size);

        // Scene edge detection
        timestamp_start = MPI_Wtime();
        compute_gradient(scene_img, grad_x, grad_y, magnitude, scene_w, scene_h, 1);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "compute_gradient (scene): %.6f s\n", timestamp_end - timestamp_start);

        timestamp_start = MPI_Wtime();
        detect_edges(magnitude, edges, scene_w, scene_h);
        timestamp_end = MPI_Wtime();
        fprintf(profile_fp, "detect_edges (scene): %.6f s\n", timestamp_end - timestamp_start);
        // if (rank == 0) save_edges_pgm("scene_edges.pgm", edges, scene_w, scene_h);

        // Iteration on each look-up table
        for (int ai = 0; ai < NUM_ANGLES; ai++) {

            Point *finalDetections = NULL;
            int detectionsCounter = 0;

            // Voting phase
            timestamp_start = MPI_Wtime();
            generalized_hough(edges, grad_x, grad_y, scene_w, scene_h, &finalDetections, &detectionsCounter, ai);
            timestamp_end = MPI_Wtime();
            fprintf(profile_fp, "generalized_hough (a=%.0f): %.6f s\n", angles[ai], timestamp_end - timestamp_start);

            // If something valid is detected -> save overlay detection
            if (detectionsCounter > 0) {

                char fname[128];
                snprintf(fname, sizeof(fname), "result_r%d_img%d_a%.0f.ppm", rank, i, angles[ai]);
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
                    int box_x = center.x - lookup_tables[ai].tw / 2;
                    int box_y = center.y - lookup_tables[ai].th / 2;
                    draw_rectangle(scene_rgb, scene_w, scene_h, box_x, box_y, lookup_tables[ai].tw, lookup_tables[ai].th, 0, 255, 0);

                    //printf("Disegno cerchio in (%d,%d) con raggio %d\n", center.x, center.y, 10);
                    //printf("Disegno rettangolo in (%d,%d) dimensioni %dx%d\n", box_x, box_y, tw, th);
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

        free(scene_img); free(grad_x); free(grad_y); free(magnitude);
        free(edges); 
    }

    // Cleanup
    free(my_paths_buffer);
    free(my_files);
    if (rank == 0) {
        for (int i = 0; i < num_files; i++) free(file_list[i]);
        free(file_list);
        free(send_counts);
        free(displs);
        free(all_paths_buffer);
        free(path_lengths);
    }

    free(lookup_tables);

    double total_end = MPI_Wtime();
    double total_duration = total_end - total_start;
    fprintf(profile_fp, "TOTAL_TIME: %.6f s\n", total_duration);

    fclose(profile_fp);

    MPI_Finalize();

    return 0;

}