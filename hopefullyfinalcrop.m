tic
NAME = 'Strio4 Originals';

f = dir(fullfile(NAME,'*.tiff')); %tif files in folder 'Originals'
num_files = length(f);
gfp_files=struct('name',{},'date',{},'bytes',{},'isdir',{},'datenum',{});
mor1_files=struct('name',{},'date',{},'bytes',{},'isdir',{},'datenum',{});
cd11_files=struct('name',{},'date',{},'bytes',{},'isdir',{},'datenum',{});

for n=1:num_files
    g=strfind(f(n).name,'GFP');
    if ~isempty(g)
        gfp_files=[gfp_files;f(n)];
    end
    m=strfind(f(n).name,'MOR1');
    if ~isempty(m)
        mor1_files=[mor1_files;f(n)];
    end
    c=strfind(f(n).name,'CD11');
    if ~isempty(c)
        cd11_files=[cd11_files;f(n)];
    end
end

files=mor1_files;
num_files=length(mor1_files);

images = cell(num_files,1); %cell to store images
image_names = cell(num_files,1); %cell to store image names

mkdir('New Code Processed Images Stage1'); %creates a new folder for processed images (used later)

%loop reads images from folder and stores the image and image name in
%respective cells
downsize = 1;  %SET AMOUNT BY WHICH TO DOWNSIZE
threshsize = 100000 %75% of the number of pixels in 2 hemispheres (to be calculated)
THRESHTYPE = 2.1;  %THRESHTYPE can be set equal to 1, 2, 2b, 3 for different options
%THRESHTYPE 1: The most common pixel brightness on the boundary will be used as the threshold
%THRESHTYPE 2: An arbitrary brightness level will be used as the threshold
%THRESHTYPE 2b: The most common pixel brightness on the boundary of pixels > 0 will be used as the threshold
%use 2.1 for mor1, use +60
%THRESHTYPE 3: The pixel N% away along the least to greatest brightness distribution will be used as the threshold
N = 50; %for THRESHTYPE 3: percentage along least to greatest brightness distribution
%Use 50 for MOR1 files for downsize images
%Want to use MOR1 files over GFP files
%Run whole code overnight with no downsizing
%tictoc
for p = 1 : num_files
    p
    filename = files(p).name;
    imgRGB = imread(fullfile(NAME,filename));    %selects image
    imgRGB = imresize(imgRGB, 1/downsize);  %downsizes image
    img = rgb2gray(imgRGB); %convert to grayscale
    [x,y] = size(img);  %determines size of image
    img_double = im2double(img);
    
    %figure;imshow(img_rescaled);
    
    if THRESHTYPE == 1
        edgepix = [];
        for i = 1:x
            edgepix = [edgepix, double(img(i,1)),double(img(i,y))];
        end
        for i = 2:y-1
            edgepix = [edgepix, double(img(1,i)), double(img(x,i))];
        end
        thresh1= mode(edgepix);
    end
    clearvars edgepix
    
    if THRESHTYPE == 2
        thresh1 = 10; %The value 10 is subject to change
    end
    
    if THRESHTYPE == 2.1
        imgtemp = img>0;
        [b2,~,n2] = bwboundaries(imgtemp);
        maxbound = 0;
        for i = 1:n2
            boundary = b2{i};
            if length(boundary) > maxbound
                maxbound = length(boundary);
            end
        end
        for i = 1:n2
            boundary = b2{i};
            if length(boundary) == maxbound
                xx = boundary(:,1);
                yy = boundary(:,2);
                boundpix = [];
                for j = 1:length(xx)
                    newpixx = xx( j);
                    newpixy = yy(j);
                    boundpix = [boundpix, double(img(newpixx, newpixy)) ];
                end
            end
        end
        thresh1 = mode(boundpix); %use for MOR1 images
        clearvars imgtemp maxbound boundary xx yy boundpix newpixx newpixy
    end
    
    if THRESHTYPE == 3
        if downsize==1 %CORRECTED 1/15/2014
            sorted = sort(img);
            a=reshape(sorted,1,x*y);
            a=sort(a);
            [xx,yy]=size(a);
            thresh1 = a(round(yy*N/100));
            %figure;imshow(i);
        else
            allpixes = [];
            for i = 1:x
                for j = 1:y
                    allpixes = [allpixes, img(i,j)];
                end
            end
            allpixes = sort(allpixes);
            [allpixx,allpixy] = size(allpixes);
            thresh1 = allpixes(round(allpixy*N/100));
        end
        clearvars allpixes allpixx allpixy
    end
    img = img > thresh1;  %converts to BW
        
    %identifying number of objects
    measurements = regionprops(img,'PixelList');
    [a,~] = size(measurements);
    all_regions = cell(1,a); %cell of all pixel matrices for regions
    numunwanted = 0;    %initializing number of unwanted regions
    for i = 1 : a
        all_regions{i} = measurements(i).PixelList;
        [m,n] = size(all_regions{i});
        if m < .1*x*y;  %determines if region in image is large
            all_regions{i} = [];
            numunwanted = numunwanted + 1;
        end
    end
    
    clearvars measurements img
    
    numobjs = a-numunwanted;    %number of large objects in the image
    new_all_regions = cell(1,numobjs);  %initializing array of large objects in image
    index1 = 1;
    for i = 1:a
        if length(all_regions{i})~= 0
            new_all_regions{index1} = all_regions{i};
            index1 = index1 + 1;
        end
    end
    
    clearvars all_regions index1 a
    
    numobjs
    %creates text file with filenames and number of regions for numobjs > 2
    if numobjs > 2
        edit more_than_2_regions.txt
        fileID = fopen('more_than_2_regions.txt','a');
        fprintf(fileID,'%s',filename, ', ');    
        fprintf(fileID, '%u\n', numobjs);   %records file with more than 2 regions of interest and number of regions of interest
        fclose(fileID);
    end
    
    %cropping for images with the correct number of regions
    if numobjs < 3
        %cropping image
        imgRGB = imread(fullfile(NAME,filename));    %selects original image
        [xxRGB,yyRGB,zzRGB] = size(imgRGB);     %determines size of original image        
        
        for k=1:numobjs
            mask = zeros(x,y);  %initializing mask of large objects
            pixel_matrix = new_all_regions{k};
            for j = 1: length(new_all_regions{k});
                mask(pixel_matrix(j,2), pixel_matrix(j,1)) = 1; %creating mask
            end
            s = regionprops(mask, 'BoundingBox');    %finds the boundary of the region
            xcrop = s(1).BoundingBox(2);    %xmin/downsize
            ycrop = s(1).BoundingBox(1);    %ymin/downsize
            width = s(1).BoundingBox(4);    %(xmax-xmin)/downsize
            height = s(1).BoundingBox(3);   %(ymax-ymin)/downsize
            
            clearvars pixel_matrix
            
            mask = mask(xcrop:xcrop+width,ycrop:ycrop+height);
            %creating the appropriate mask and finding orientation
            %if numobjs == 1
            %    mask = mask(xcrop:xcrop+width, ycrop:ycrop+height); %cropping mask for 1 object
            %elseif numobjs == 2
            %    mask = mask(xcrop:xcrop+width, ycrop:ycrop+height);   %cropping mask for 2 objects
                
            %end
            
            %altering mask so that the region is filled
            [xd,yd] = size(mask);
            [b,~,NN] = bwboundaries(mask);  %determines boundaries of objects in mask, and number of parent boundaries
            max2 = 0;
            for i = (1:NN)
                boundary = b{i};
                if length(boundary) > max2
                    max2 = length(boundary);   %ensures we only select the largest parent boundary
                end
            end
            for i = (1:NN)
                boundary = b{i};
                if length(boundary) == max2
                    xx = boundary(:,1);
                    yy = boundary(:,2);
                    boundaryonly = zeros(xd,yd);  %matrix of just the boundary pixels
                    for i = 1:length(xx)
                        boundaryonly(xx(i), yy(i)) = 1;
                    end
                    mask = poly2mask(xx,yy,yd,xd);
                    mask = imrotate(mask, 270);
                    mask = flipdim(mask,2);
                    mask = mask + boundaryonly;
                    mask = (mask~=0);   %finalizes mask
                    o = regionprops(mask,'Orientation'); %determines orientation of the object
                    mask = imrotate(mask, -1*o(1).Orientation+180); %rotating mask
                end
            end
            
            if numobjs == 2
                mask = bwareaopen(mask, round(.1*x*y));  %completed template for two separate hemispheres (template is cropped and rotated)
            end


            clearvars boundaryonly boundary xx yy max2 b NN
            
            %finding center of object
            centroid = regionprops(mask, 'Centroid');
            [jkl,~] = size(centroid);
            centxest = xd;
            for i = (1:jkl)
                if abs(centroid(i).Centroid(1) - xd/2) < centxest;  %choosing centroid closest to center of image (if more than one centroid)
                    centxest = abs(centroid(i).Centroid(1)-xd/2);
                    centy = centroid(i).Centroid(1);
                    centx = centroid(i).Centroid(2);
                end
            end
            
            clearvars jkl centxest centroid
            
            imgRGB = imgRGB(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);   %cropping original image
            imgRGB = imrotate(imgRGB, -1*o(1).Orientation+180); %rotating original image
            [newx,newy,newz] = size(imgRGB);
            
%             gfp_name = gfp_files(p).name;
%             gfp_file = imread(fullfile(NAME,gfp_name));
%             gfp_file = gfp_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
%             gfp_file=imrotate(gfp_file,-1*o(1).Orientation+180);
%             cd11_name = cd11_files(p).name;
%             cd11_file = imread(fullfile(NAME,cd11_name));
%             cd11_file = cd11_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
%             cd11_file = imrotate(cd11_file,-1*o(1).Orientation+180);
            
            %saving for 2 objects
            if numobjs == 2
                gfp_name = gfp_files(p).name;
                cd11_name = cd11_files(p).name;
                if k == 1   %determines if hemisphere is left
                    newnameone = strrep(filename, '.tiff','_L_processed1.tiff');
                    newnametwo = strrep(gfp_name, '.tiff','_L_processed1.tiff');
                    newnamethree = strrep(cd11_name, '.tiff','_L_processed1.tiff');
                elseif k == 2   %determines if hemisphere is right
                    newnameone = strrep(filename, '.tiff','_R_processed1.tiff');
                    newnametwo = strrep(gfp_name, '.tiff','_R_processed1.tiff');
                    newnamethree = strrep(cd11_name, '.tiff','_R_processed1.tiff');
                end
                mask=cat(3,mask,mask,mask);

                imgRGB(~mask) = 0;
                imwrite(imgRGB,['New Code Processed Images Stage1/', newnameone], 'tiff');
                clearvars imgRGB newnameone
                 
                gfp_file = imread(fullfile(NAME,gfp_name));
                gfp_file = gfp_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                gfp_file=imrotate(gfp_file,-1*o(1).Orientation+180);
                gfp_file(~mask) = 0;
                imwrite(gfp_file,['New Code Processed Images Stage1/', newnametwo], 'tiff');
                clearvars gfp_file newnametwo gfp_name
                
                cd11_file = imread(fullfile(NAME,cd11_name));
                cd11_file = cd11_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                cd11_file = imrotate(cd11_file,-1*o(1).Orientation+180);
                cd11_file(~mask) = 0;
                clearvars mask
                imwrite(cd11_file,['New Code Processed Images Stage1/', newnamethree], 'tiff');
                clearvars cd11_file newnamethree cd11_name
                
                if k == 1
                    imgRGB = imread(fullfile(NAME,filename));
                end

            end
            
            %splitting and saving for 1 object
            if numobjs == 1
                numpixones = regionprops(mask, 'Area');
                a=max([numpixones(:).Area]); %CORRECTED 1/22/2014;
                clearvars tempimg
                if a > threshsize
                    %finding dividing line: first specialLine for 1/downsize,
                    %then largespecialLine for the original image
                    
                    %creating specialLine
                    thresh = 985/downsize;    %SET A THRESHOLD
                    newbox = mask(:,round(centy-thresh):round(centy+thresh));
                    [newx1, newy1] = size(newbox);
                    %newbox = rot90(rot90(newbox));
                    specialLine = zeros(newx1,1);    %initializing matrix of median of location of 0's in each row of newbox
                    for i = 1:newx1
                        list = [];
                        for j = 1:newy1
                            if newbox(i,j) == 0
                                list = [list,j];
                            end
                        end
                        if size(list) ~= [0,0]
                            specialLine(i) = median(list)+centy - thresh;   %specialLine is complete
                        end
                    end
                    
                    clearvars newbox list centy centx
                    

                    
                    %creating largespecialLine
                    largespecialLine = zeros(newx,1);   %initialize
                    for i = 1:newx
                        largespecialLine(i) = specialLine(ceil(i/downsize))*newx/newx1;
                    end
                    %filling in holes in largespecialLine using the y = mx+b
                    for i = 1:newx-1
                        if largespecialLine(i) ~= 0
                            if largespecialLine(i+1) == 0
                                Top = largespecialLine(i);  %determines where there are 0's in largespecialLine
                                L = 0;
                                j=i+1;
                                while L == 0
                                    if largespecialLine(j) ~= 0 && i ~=newx %CORRECTED 1/15/2014
                                        L = 1;
                                        Bot = largespecialLine(j);
                                    end
                                    j = j+1;
                                end
                                slope = (Top-Bot)/(i-j+3);  %determines slope for y = mx+b
                                intercept = Top -(slope)*(i+1); %determines intercept for y = mx+b
                                for L = i+1:j-2
                                    largespecialLine(L) = slope*L + intercept;  %largespecialLine complete
                                end
                            end
                        end
                    end
                    clearvars slope intercept L i Top j Bot
                    
%                     figure;imshow(mask)
%                     hold on
%                     plot(specialLine,[1:8769],'Color','r','LineWidth',2)
%                     
%                     figure,imshow(imgRGB)
%                     hold on
%                     plot(repmat(centy-thresh,1,10001),[0:10000],'Color','r','LineWidth',2)
%                     hold on
%                     plot(repmat(centy+thresh,1,10001),[0:10000],'Color','r','LineWidth',2)
%                     plot(largespecialLine,[1:8769],'Color','r','LineWidth',2)
                    
                    
                    maskcopy = mask;

                    %creating and saving left image
                    imgRGBL = imgRGB;
                    for i = 1:newx
                        for j = 1:newy
                            if j > largespecialLine(i)
                                mask(i,j,:) = 0;
                            end
                        end
                    end
                    for i = 1:newx
                        for j = 1:newy
                            if j < largespecialLine(i)
                                maskcopy(i,j,:) = 0;
                            end
                        end
                    end
                    mask = bwareaopen(mask, round(.1*x*y));
                    mask=cat(3,mask,mask,mask);
                    maskcopy = bwareaopen(maskcopy, round(.1*x*y));
                    maskcopy = cat(3, maskcopy, maskcopy, maskcopy);
                    
                    imgRGBL(~mask) = 0;
                    imgRGBL = imgRGBL(:, 1:max(largespecialLine),:);
                    
                    imgRGBL=background9000_L(imgRGBL);
                    
                    newname = strrep(filename, '.tiff','_L_processed1.tiff');
                    imwrite(imgRGBL,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars imgRGBL
                    imgRGB(~maskcopy) = 0;
                    imgRGB = imgRGB(:,min(largespecialLine):newy,:);
                    
                    imgRGB=background9000_R(imgRGB);
                    
                    newname = strrep(filename, '.tiff','_R_processed1.tiff');
                    imwrite(imgRGB,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars imgRGB filename
                    
                    gfp_name = gfp_files(p).name;
                    gfp_file = imread(fullfile(NAME,gfp_name));
                    gfp_file = gfp_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                    gfp_file=imrotate(gfp_file,-1*o(1).Orientation+180);
                    gfp_file_L = gfp_file;
                    
                    gfp_file_L(~mask) = 0;
                    gfp_file_L = gfp_file_L(:, 1:max(largespecialLine),:);
                    
                    gfp_file_L=background9000_L(gfp_file_L);
                    
                    newname = strrep(gfp_name, '.tiff','_L_processed1.tiff');
                    imwrite(gfp_file_L,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars gfp_file_L
                    gfp_file(~maskcopy) = 0;
                    gfp_file = gfp_file(:,min(largespecialLine):newy,:);
                    
                    gfp_file=background9000_R(gfp_file);
                    
                    newname = strrep(gfp_name, '.tiff','_R_processed1.tiff');
                    imwrite(gfp_file,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars gfp_file gfp_name

                    cd11_name = cd11_files(p).name;
                    cd11_file = imread(fullfile(NAME,cd11_name));
                    cd11_file = cd11_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                    cd11_file = imrotate(cd11_file,-1*o(1).Orientation+180);
                    cd11_file_L = cd11_file;
                    
                    cd11_file_L(~mask) = 0;
                    cd11_file_L = cd11_file_L(:,1:max(largespecialLine),:);
                    
                    cd11_file_L=background9000_L(cd11_file_L);
                    
                    newname = strrep(cd11_name, '.tiff','_L_processed1.tiff');
                    imwrite(cd11_file_L,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars cd11_file_L mask
                    cd11_file(~maskcopy) = 0;
                    cd11_file = cd11_file(:,min(largespecialLine):newy,:);
                    
                    cd11_file=background9000_R(cd11_file);
                    
                    newname = strrep(cd11_name, '.tiff','_R_processed1.tiff');
                    imwrite(cd11_file,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars cd11_file_R maskcopy cd11_name newname largeSpecialLine
                else
                    mask = bwareaopen(mask, round(.1*x*y));
                    mask=cat(3,mask,mask,mask);
                    imgRGB(~mask) = 0;
                    

                    
                    newname = strrep(filename, '.tiff', '_LR_processed1.tiff');
                    imwrite(imgRGB,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars imgRGB
                    
                    gfp_name = gfp_files(p).name;
                    gfp_file = imread(fullfile(NAME,gfp_name));
                    gfp_file = gfp_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                    gfp_file=imrotate(gfp_file,-1*o(1).Orientation+180);
                    gfp_file(~mask) = 0;
                    

                    
                    newname = strrep(gfp_name, '.tiff','_LR_processed1.tiff');
                    imwrite(gfp_file,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars gfp_file
                    
                    cd11_name = cd11_files(p).name;
                    cd11_file = imread(fullfile(NAME,cd11_name));
                    cd11_file = cd11_file(xcrop*downsize:(xcrop+width)*downsize, ycrop*downsize:(ycrop+height)*downsize,:);
                    cd11_file = imrotate(cd11_file,-1*o(1).Orientation+180);
                    cd11_file(~mask) = 0;
                    

                    
                    newname = strrep(cd11_name, '.tiff','_LR_processed1.tiff');
                    imwrite(cd11_file,['New Code Processed Images Stage1/', newname], 'tiff');
                    clearvars cd11_file mask filename gfp_name cd11_name newname
                end
            end
        end
    end
    clearvars new_all_regions
end
toc
