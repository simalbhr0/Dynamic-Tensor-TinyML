#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ============================= */
/*         TENSOR TYPE           */
/* ============================= */

typedef enum {
    TENSOR_FLOAT32,
    TENSOR_FLOAT16,
    TENSOR_INT8
} TensorType;

/* ============================= */
/*         TENSOR STRUCT         */
/* ============================= */

typedef struct {
    TensorType type;
    uint16_t rows;
    uint16_t cols;
    union {
        float *f32;
        uint16_t *f16;
        int8_t *i8;
    } data;
} Tensor;

/* ============================= */
/*     FLOAT32 -> FLOAT16        */
/* ============================= */

uint16_t float32_to_float16(float value)
{
    uint32_t bits;
    uint16_t sign;
    int16_t exponent;
    uint16_t mantissa;

    bits = *((uint32_t*)&value);

    sign = (bits >> 16) & 0x8000;
    exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    mantissa = (bits >> 13) & 0x3FF;

    if (exponent <= 0)
        return 0;

    if (exponent >= 31)
        return sign | 0x7C00;

    return sign | (exponent << 10) | mantissa;
}

float float16_to_float32(uint16_t value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    uint32_t bits;

    if ((value & 0x7FFF) == 0)
        return 0.0f;

    sign = (value & 0x8000) << 16;
    exponent = ((value >> 10) & 0x1F);
    mantissa = (value & 0x3FF);

    exponent = exponent - 15 + 127;

    bits = sign | (exponent << 23) | (mantissa << 13);

    return *((float*)&bits);
}

/* ============================= */
/*        CREATE TENSOR          */
/* ============================= */

Tensor* create_tensor(uint16_t rows, uint16_t cols, TensorType type)
{
    Tensor *t;
    uint32_t total;

    t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL)
        return NULL;

    t->rows = rows;
    t->cols = cols;
    t->type = type;

    total = rows * cols;

    switch(type)
    {
        case TENSOR_FLOAT32:
            t->data.f32 = (float*)malloc(total * sizeof(float));
            break;

        case TENSOR_FLOAT16:
            t->data.f16 = (uint16_t*)malloc(total * sizeof(uint16_t));
            break;

        case TENSOR_INT8:
            t->data.i8 = (int8_t*)malloc(total * sizeof(int8_t));
            break;
    }

    return t;
}

/* ============================= */
/*         FREE TENSOR           */
/* ============================= */

void free_tensor(Tensor *t)
{
    if (t == NULL)
        return;

    switch(t->type)
    {
        case TENSOR_FLOAT32:
            free(t->data.f32);
            break;

        case TENSOR_FLOAT16:
            free(t->data.f16);
            break;

        case TENSOR_INT8:
            free(t->data.i8);
            break;
    }

    free(t);
}

/* ============================= */
/*         PRINT TENSOR          */
/* ============================= */

void print_tensor(Tensor *t)
{
    uint32_t total;
    uint32_t i;

    total = t->rows * t->cols;

    printf("Tensor (%dx%d):\n", t->rows, t->cols);

    for (i = 0; i < total; i++)
    {
        if (t->type == TENSOR_FLOAT32)
            printf("%.3f ", t->data.f32[i]);
        else if (t->type == TENSOR_FLOAT16)
            printf("%.3f ", float16_to_float32(t->data.f16[i]));
        else if (t->type == TENSOR_INT8)
            printf("%d ", t->data.i8[i]);

        if ((i + 1) % t->cols == 0)
            printf("\n");
    }
}

/* ============================= */
/*        QUANTIZATION           */
/* ============================= */

void quantize_float32_to_int8(Tensor *src, Tensor *dst, float scale)
{
    uint32_t total;
    uint32_t i;
    float value;

    if (src->type != TENSOR_FLOAT32 || dst->type != TENSOR_INT8)
        return;

    total = src->rows * src->cols;

    for (i = 0; i < total; i++)
    {
        value = src->data.f32[i] / scale;

        if (value > 127)
            value = 127;
        if (value < -128)
            value = -128;

        dst->data.i8[i] = (int8_t)value;
    }
}

/* ============================= */
/*       DEQUANTIZATION          */
/* ============================= */

void dequantize_int8_to_float32(Tensor *src, Tensor *dst, float scale)
{
    uint32_t total;
    uint32_t i;

    if (src->type != TENSOR_INT8 || dst->type != TENSOR_FLOAT32)
        return;

    total = src->rows * src->cols;

    for (i = 0; i < total; i++)
    {
        dst->data.f32[i] = src->data.i8[i] * scale;
    }
}

/* ============================= */
/*      MEMORY COMPARISON        */
/* ============================= */

void print_memory_usage(uint32_t elements)
{
    printf("\nMemory Usage Comparison (%u elements):\n", elements);
    printf("Float32 : %lu bytes\n", (unsigned long)(elements * sizeof(float)));
    printf("Float16 : %lu bytes\n", (unsigned long)(elements * sizeof(uint16_t)));
    printf("Int8    : %lu bytes\n", (unsigned long)(elements * sizeof(int8_t)));
}

/* ============================= */
/*              MAIN             */
/* ============================= */

int main()
{
    Tensor *input;
    Tensor *quantized;
    Tensor *dequantized;

    printf("=== Dynamic Tensor Demo ===\n\n");

    input = create_tensor(2, 2, TENSOR_FLOAT32);

    input->data.f32[0] = 0.5f;
    input->data.f32[1] = -1.2f;
    input->data.f32[2] = 3.4f;
    input->data.f32[3] = 2.1f;

    printf("Original Float32 Tensor:\n");
    print_tensor(input);

    quantized = create_tensor(2, 2, TENSOR_INT8);
    quantize_float32_to_int8(input, quantized, 0.1f);

    printf("\nQuantized INT8 Tensor:\n");
    print_tensor(quantized);

    dequantized = create_tensor(2, 2, TENSOR_FLOAT32);
    dequantize_int8_to_float32(quantized, dequantized, 0.1f);

    printf("\nDequantized Back To Float32:\n");
    print_tensor(dequantized);

    print_memory_usage(4);

    free_tensor(input);
    free_tensor(quantized);
    free_tensor(dequantized);

    return 0;
}
