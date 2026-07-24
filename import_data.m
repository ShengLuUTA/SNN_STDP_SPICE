% % import data
% clear; file = 'D:\temp\temp.lis';
function [data, name] = import_data(file)
group = 0; in = 1;
fid = fopen(file); alldata = []; top = 1; final = [];
while(1)
    tline = fgets(fid);
    if tline(1)==-1; break; end;
% % % % 2d sweep
%     if strfind(tline,' *** ')  
%         if exist('num')
%             alldata = [alldata; [num*ones(size(datagroup,1),1) datagroup]]; 
%         end
%         take = regexp(tline,'parameter (\S+)\s+=\s+(\S+)\s','tokens');
%         num = str2num(take{1}{2}); num_name = take{1}(1);
%         
%         datagroup = []; in = 1; top = 0; group = 0;
%     end
    if tline(1)=='x'
        i = 1;
        fgets(fid);fgets(fid);
        tline = fgets(fid);
        temp = textscan(tline,'%s');
        for ii = 1:length(temp{1})
            name(in) = temp{1}(ii); in = in+1;
        end
        while(1)
            tline = fgets(fid);
            if tline(1)=='y'; 
                break; 
            end
            temp = sscanf(tline,'%f')';
            tt(i) = temp(1);
            datagroup(i,:) = temp;
%             datagroup(i,1) = temp(1);
%             for j = 2:length(temp)
%                 datagroup(i,1) = temp(1);
%                 datagroup(i,group*4+j) = temp(j);
%             end
            i = i+1;
        end
%         group = group+1;
        final = [final datagroup(:,2:end)]; datagroup = [];
    end
%     final = [final datagroup(:,2:end)]; datagroup = [];
end
final = [tt' final];
if isempty(alldata) && ~exist('datagroup')    % aborted
    data = []; name = [];
else
    if ~isempty(alldata)
        name = [num_name {'sweep'} name];
        data = alldata; 
    else
        name = [{'sweep'} name];
        data = final;
    end
end

fclose(fid);







