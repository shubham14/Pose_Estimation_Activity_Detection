function lines = hough_lines_out(im, inp)
   im = rgb2gray(imread(im));
   BW = edge(im, 'canny');
   if inp == 'L'
       [H,T,R] = hough(BW,'Theta', 30:0.5:70);
   else
       [H,T,R] = hough(BW,'Theta', -40:-0.5:-70);
   end
   P  = houghpeaks(H,1,'threshold',ceil(0.3*max(H(:))));
   lines = houghlines(BW,T,R,P);
%    figure, imshow(im), hold on
   max_len = 0;
   for k = 1:length(lines)
      xy = [lines(k).point1; lines(k).point2];
%       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

      % Plot beginnings and ends of lines
%       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

      % Determine the endpoints of the longest line segment
      len = norm(lines(k).point1 - lines(k).point2);
      if ( len > max_len)
         max_len = len;
         xy_long = xy;
      end
  end