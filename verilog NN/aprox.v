module aprox(out, in);
    output reg signed [31:0] out;
    input wire signed [31:0] in;

    wire signed [31:0] one = 65536;
    wire signed [31:0] two = 131072;
    wire signed [31:0] oneThird = 21845;
    wire signed [31:0] twoThird = 43695;
    wire signed [31:0] maxlow = 1;
    wire signed [31:0] slightOneThird = 21844;
    
    always @* begin
        if (in < one)
            out <= -in+one;
        else begin
            if (in > two)
                out <= -maxlow*in+slightOneThird;
            else
                out <= -oneThird*in+twoThird;
        end 
    end

endmodule