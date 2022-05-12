module neuron(out_val, b, x1, w1, x2, w2);
    output wire signed [31:0] out_val;
    input wire signed [31:0] b, x1, w1, x2, w2;

    //for fixed point we want 1 for sign 8 bits are for number  23 for fraction
    wire [31:0] z; 
    wire [31:0] abs_z; 
    wire signed [31:0] e = 178126;
    wire signed [31:0] one = 65536;
    wire signed [31:0] onehalf = 32768;
    assign z = b+(x1*w1)+(x2*w2);
    abs absz(abs_z, z);
    assign out_val  = onehalf*(one+(z/(one+abs_z));
endmodule