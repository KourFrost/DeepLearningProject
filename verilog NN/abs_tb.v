module abs_tb();

    reg signed [31:0] b= -32768;
   
    wire signed [31:0] out_val;
  

    abs a(out_val, b);

    initial
     $monitor("At time %t, out:%f b:%f ", $time, out_val, b);
endmodule