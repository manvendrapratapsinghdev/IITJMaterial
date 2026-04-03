/*
 * First Convolutional Layer of LeNet-5
 * 
 * Input: 28x28x1 image (grayscale)
 * Kernel: 5x5, 6 filters
 * Stride: 1
 * Padding: 2
 * Output: 28x28x6 feature maps
 * 
 * Fixed-point format: Q8.8 (8 integer bits, 8 fractional bits)
 */

module conv2d_layer1 #(
    parameter INPUT_WIDTH = 28,
    parameter INPUT_HEIGHT = 28,
    parameter KERNEL_SIZE = 5,
    parameter NUM_FILTERS = 6,
    parameter PADDING = 2,
    parameter DATA_WIDTH = 16  // Q8.8 fixed point
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] pixel_in,
    input wire pixel_valid,
    
    output reg signed [DATA_WIDTH-1:0] feature_out,
    output reg feature_valid,
    output reg done
);

    // Padded dimensions
    localparam PADDED_WIDTH = INPUT_WIDTH + 2*PADDING;
    localparam PADDED_HEIGHT = INPUT_HEIGHT + 2*PADDING;
    
    // State machine
    localparam IDLE = 3'd0;
    localparam LOAD_INPUT = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam OUTPUT = 3'd3;
    localparam DONE = 3'd4;
    
    reg [2:0] state;
    
    // Input buffer (padded)
    reg signed [DATA_WIDTH-1:0] input_buffer [0:PADDED_HEIGHT-1][0:PADDED_WIDTH-1];
    
    // Weight memory (6 filters x 5x5 kernel)
    reg signed [DATA_WIDTH-1:0] weights [0:NUM_FILTERS-1][0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];
    
    // Bias memory
    reg signed [DATA_WIDTH-1:0] biases [0:NUM_FILTERS-1];
    
    // Output buffer
    reg signed [DATA_WIDTH-1:0] output_buffer [0:NUM_FILTERS-1][0:INPUT_HEIGHT-1][0:INPUT_WIDTH-1];
    
    // Counters
    reg [9:0] load_row, load_col;
    reg [9:0] out_row, out_col;
    reg [2:0] filter_idx;
    reg [2:0] k_row, k_col;
    
    // Accumulator for convolution
    reg signed [31:0] accumulator;
    
    integer i, j, k;
    
    // Initialize weights and biases (would be loaded from memory in real design)
    initial begin
        // Initialize to zero (in practice, load from file)
        for (i = 0; i < NUM_FILTERS; i = i + 1) begin
            biases[i] = 0;
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                for (k = 0; k < KERNEL_SIZE; k = k + 1) begin
                    weights[i][j][k] = 0;
                end
            end
        end
    end
    
    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            feature_valid <= 0;
            load_row <= 0;
            load_col <= 0;
            out_row <= 0;
            out_col <= 0;
            filter_idx <= 0;
            
            // Clear input buffer (padding with zeros)
            for (i = 0; i < PADDED_HEIGHT; i = i + 1) begin
                for (j = 0; j < PADDED_WIDTH; j = j + 1) begin
                    input_buffer[i][j] <= 0;
                end
            end
            
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    feature_valid <= 0;
                    if (start) begin
                        state <= LOAD_INPUT;
                        load_row <= PADDING;
                        load_col <= PADDING;
                    end
                end
                
                LOAD_INPUT: begin
                    if (pixel_valid) begin
                        input_buffer[load_row][load_col] <= pixel_in;
                        
                        if (load_col == INPUT_WIDTH + PADDING - 1) begin
                            load_col <= PADDING;
                            if (load_row == INPUT_HEIGHT + PADDING - 1) begin
                                state <= COMPUTE;
                                out_row <= 0;
                                out_col <= 0;
                                filter_idx <= 0;
                            end else begin
                                load_row <= load_row + 1;
                            end
                        end else begin
                            load_col <= load_col + 1;
                        end
                    end
                end
                
                COMPUTE: begin
                    // Perform convolution for current output position
                    accumulator = biases[filter_idx];
                    
                    for (k_row = 0; k_row < KERNEL_SIZE; k_row = k_row + 1) begin
                        for (k_col = 0; k_col < KERNEL_SIZE; k_col = k_col + 1) begin
                            accumulator = accumulator + 
                                (input_buffer[out_row + k_row][out_col + k_col] * 
                                 weights[filter_idx][k_row][k_col]) >>> 8;  // Shift for Q8.8
                        end
                    end
                    
                    // ReLU activation
                    if (accumulator < 0)
                        output_buffer[filter_idx][out_row][out_col] <= 0;
                    else if (accumulator > 32'h7FFF)
                        output_buffer[filter_idx][out_row][out_col] <= 16'h7FFF;
                    else
                        output_buffer[filter_idx][out_row][out_col] <= accumulator[15:0];
                    
                    // Move to next position
                    if (filter_idx == NUM_FILTERS - 1) begin
                        filter_idx <= 0;
                        if (out_col == INPUT_WIDTH - 1) begin
                            out_col <= 0;
                            if (out_row == INPUT_HEIGHT - 1) begin
                                state <= OUTPUT;
                                out_row <= 0;
                                out_col <= 0;
                                filter_idx <= 0;
                            end else begin
                                out_row <= out_row + 1;
                            end
                        end else begin
                            out_col <= out_col + 1;
                        end
                    end else begin
                        filter_idx <= filter_idx + 1;
                    end
                end
                
                OUTPUT: begin
                    feature_out <= output_buffer[filter_idx][out_row][out_col];
                    feature_valid <= 1;
                    
                    if (filter_idx == NUM_FILTERS - 1) begin
                        filter_idx <= 0;
                        if (out_col == INPUT_WIDTH - 1) begin
                            out_col <= 0;
                            if (out_row == INPUT_HEIGHT - 1) begin
                                state <= DONE;
                            end else begin
                                out_row <= out_row + 1;
                            end
                        end else begin
                            out_col <= out_col + 1;
                        end
                    end else begin
                        filter_idx <= filter_idx + 1;
                    end
                end
                
                DONE: begin
                    done <= 1;
                    feature_valid <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
