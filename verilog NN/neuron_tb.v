module neuron_tb();

    reg signed [31:0] b= 32768;
    reg signed [31:0] x1= 32768;
    reg signed [31:0] w1= 32768;
    reg signed [31:0] x2= -32768;
    reg signed [31:0] w2= 22937;
   
    wire signed [31:0] out_val;
  
    // initial
    // begin
    //     #15;
    //     b= 2.7;
    //     x1= 0.5;
    //     w1= 0.23;
    //     x2= 0.45;
    //     w2= 0.00544;
    // end

    neuron n(out_val, b, x1, w1, x2, w2);

    initial
     $monitor("At time %t, out:%f b:%f x1:%f w1:%f x2:%f w2:%f", $time, out_val, b, x1, w1, x2, w2);
endmodule