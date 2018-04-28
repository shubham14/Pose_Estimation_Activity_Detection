function proc_image(img_loc,out_loc)
% img_loc ='sample_images/thumbnail001.jpg';
img = imread(img_loc);

    linesR = hough_lines_out(img_loc,'R');
    linesL = hough_lines_out(img_loc,'L');

    p1R = [linesR.point1];
    p2R = [linesR.point2];

    xR = [p1R(1:2:end) p2R(1:2:end)];
    yR = [p1R(2:2:end) p2R(2:2:end)];

    pR = polyfit(xR,yR,1); 
    fR = polyval(pR,xR); 

    p1L = [linesL.point1];
    p2L = [linesL.point2];

    xL = [p1L(1:2:end) p2L(1:2:end)];
    yL = [p1L(2:2:end) p2L(2:2:end)];

    pL = polyfit(xL,yL,1); 
    fL = polyval(pL,xL); 
    
    x_inter = (pL(2)-pR(2))/(pR(1)-pL(1));
    y_inter = pL(1)*x_inter + pL(2);
    
    

    % disp([x_inter y_inter])
    % disp(pL)
    % disp(pR)                            
% figure, imshow(img), hold on
% 
% xl = linspace(xL(1),x_inter,100);
% yl = linspace(fL(1),y_inter,100);
% 
% xr = linspace(x_inter,xR(end),100);
% yr = linspace(y_inter,fR(end),100);
% 
% plot(xl,yl,'LineWidth',2,'Color','green');
% plot(xr,yr,'LineWidth',2,'Color','blue');
middle1 = (0.5*xL(1)+1.5*x_inter)/2;
middle2 = (0.5*xR(end)+1.5*x_inter)/2;

const = 10;
[n,m,c] = size(img);
for y=1:n
    for x=1:m
        if( ((y-pL(1)*x-pL(2)-const) > 0 ) && ((y-pR(1)*x-pR(2)-const)>0)  || ( middle1<=x && x<= middle2))
            img(y,x,:) = [0,0,0];
        end
    end
end
imwrite(img,out_loc)
end


