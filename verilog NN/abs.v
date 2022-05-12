module abs(out, in);
    output wire signed [31:0] out;
    input wire signed [31:0] in;

    assign out = in[31] ? -in : in ;

endmodule